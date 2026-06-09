"""Split-K (bf16 atomic) correctness + perf for dense mxfp4 on low-occupancy
shapes (kv N=1024). ref = split_k=1; compare SNR + TFLOPS for split_k in {2,4,8}."""
import sys, os, torch
_R = "/workspace/code/FlyDSL"
for p in (_R, _R + "/flydsl/src"):
    sys.path.insert(0, p)
import flydsl.compiler as flyc
from turbo.mxfp4_gemm_8wave import compile_mxfp4_gemm_8w
from turbo.mxfp8_gemm_8wave import preshuffle_scale, preshuffle_scale_b_comb
SB = 32


def bench(fn, it=30, wu=8, reps=4):
    fn(); torch.cuda.synchronize()
    for _ in range(wu):
        fn()
    torch.cuda.synchronize(); best = 1e9
    for _ in range(reps):
        e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(); e0.record()
        for _ in range(it):
            fn()
        e1.record(); torch.cuda.synchronize(); best = min(best, e0.elapsed_time(e1) * 1000 / it)
    return best


def snr(o, r):
    o = o.float(); r = r.float()
    return 10 * torch.log10((r ** 2).sum() / ((o - r) ** 2).sum().clamp_min(1e-20)).item()


def make(M, N, K):
    d = "cuda"
    a = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device=d)
    b = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=d)
    asc = torch.randint(125, 130, (M, K // SB), dtype=torch.uint8, device=d)
    bsc = torch.randint(125, 130, (N, K // SB), dtype=torch.uint8, device=d)
    asp = preshuffle_scale(asc, K, 4); bsp = preshuffle_scale_b_comb(bsc, K)
    return a.view(torch.int8).view(-1), b.view(torch.int8).view(-1), asp.view(-1), bsp.view(-1)


def run(M, N, K, G, inp):
    d = "cuda"
    ai, bi, asp, bsp = inp; st = torch.cuda.current_stream()
    # split_k>1 atomic-adds FP32 into a scratch buffer, then host casts -> bf16.
    cdt = torch.float32 if G > 1 else torch.bfloat16
    c = torch.zeros((M, N), dtype=cdt, device=d)
    throwaway = torch.zeros_like(c)
    fn = compile_mxfp4_gemm_8w(K=K, BLOCK_M=256, BLOCK_N=256, mode="pipe", split_k=G)
    cc = flyc.compile(fn, ai, bi, throwaway.view(-1), asp.view(-1), bsp.view(-1), M, N, st)
    torch.cuda.synchronize()
    c.zero_()
    cc(ai, bi, c.view(-1), asp.view(-1), bsp.view(-1), M, N, st)
    torch.cuda.synchronize()
    out = c.to(torch.bfloat16) if G > 1 else c.clone()

    if G > 1:
        def call():
            c.zero_()
            cc(ai, bi, c.view(-1), asp.view(-1), bsp.view(-1), M, N, st)
            return c.to(torch.bfloat16)
    else:
        def call():
            cc(ai, bi, c.view(-1), asp.view(-1), bsp.view(-1), M, N, st)
    us = bench(call)
    tf = 2 * M * N * K / (us / 1e6) / 1e12
    return out, tf, us


for tag, M, N, K in [("70B kv", 4096, 1024, 8192), ("70B kv", 8192, 1024, 8192)]:
    inp = make(M, N, K)
    ref, tf1, us1 = run(M, N, K, 1, inp)
    print(f"{tag} M={M} N={N} K={K}: split_k=1 {tf1:5.0f} TF ({us1:.1f}us)")
    for G in (2, 4, 8):
        out, tf, us = run(M, N, K, G, inp)
        s = snr(out, ref)
        print(f"    split_k={G}: {tf:5.0f} TF ({us:.1f}us)  SNR={s:5.1f}dB  {tf/tf1:.2f}x")
