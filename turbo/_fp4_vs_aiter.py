"""Side-by-side: FlyDSL dense mxfp4 (no weight preshuffle, only scale) vs aiter
gemm_a4w4 (hand-asm, B preshuffled) at Llama 7B/70B shapes, M in {4096,8192}.
Reports TFLOPS for competitor and for mine (gn=0 baseline + best band swizzle)."""
import sys, os, torch
_R = "/workspace/code/FlyDSL"
for p in (_R, _R + "/flydsl/src"):
    sys.path.insert(0, p)
import aiter
from aiter.ops.shuffle import shuffle_weight
import flydsl.compiler as flyc
from turbo.mxfp4_gemm_8wave import compile_mxfp4_gemm_8w
from turbo.mxfp8_gemm_8wave import preshuffle_scale, preshuffle_scale_b_comb
SB = 32
quant_func = aiter.get_triton_quant(aiter.QuantType.per_1x32)


def bench(fn, it=30, wu=10, reps=4):
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


def comp_tf(M, N, K):
    x = torch.randn((M, K), dtype=torch.bfloat16, device="cuda")
    w = torch.randn((N, K), dtype=torch.bfloat16, device="cuda")
    xq, xs = quant_func(x, shuffle=True)
    wq, ws = quant_func(w, shuffle=True)
    wsh = shuffle_weight(wq, layout=(16, 16))
    fn = lambda: aiter.gemm_a4w4(xq, wsh, xs, ws, bpreshuffle=True)
    fn()
    us = bench(fn)
    return 2 * M * N * K / (us / 1e6) / 1e12


def mine_tf(M, N, K, gn):
    d = "cuda"
    a = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device=d)
    b = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=d)
    asc = torch.randint(125, 130, (M, K // SB), dtype=torch.uint8, device=d)
    bsc = torch.randint(125, 130, (N, K // SB), dtype=torch.uint8, device=d)
    c = torch.zeros((M, N), dtype=torch.bfloat16, device=d)
    asp = preshuffle_scale(asc, K, 4); bsp = preshuffle_scale_b_comb(bsc, K)
    ai = a.view(torch.int8).view(-1); bi = b.view(torch.int8).view(-1); st = torch.cuda.current_stream()
    fn = compile_mxfp4_gemm_8w(K=K, BLOCK_M=256, BLOCK_N=256, mode="pipe", group_m=4, group_n=gn)
    ar = (ai, bi, c.view(-1), asp.view(-1), bsp.view(-1), M, N, st)
    cc = flyc.compile(fn, *ar); cc(*ar); torch.cuda.synchronize()
    us = bench(lambda: cc(*ar))
    return 2 * M * N * K / (us / 1e6) / 1e12


SHAPES = [
    ("7B q/o", 4096, 4096), ("7B gate/up", 11008, 4096), ("7B down", 4096, 11008),
    ("70B q/o", 8192, 8192), ("70B kv", 1024, 8192),
    ("70B gate/up", 28672, 8192), ("70B down", 8192, 28672),
]
print(f"{'shape':12s} {'M':>5s} {'N':>5s} {'K':>5s}  {'aiter':>6s} {'fly0':>6s} {'flyGN':>6s} {'gn':>3s}  fly/aiter")
for M in (4096, 8192):
    print(f"--- M={M} ---")
    for tag, N, K in SHAPES:
        nb = N // 256
        ct = comp_tf(M, N, K)
        f0 = mine_tf(M, N, K, 0)
        gns = sorted({g for g in (max(1, nb // 8), max(1, nb // 4)) if g < nb})
        fbest, gbest = f0, 0
        for gn in gns:
            t = mine_tf(M, N, K, gn)
            if t > fbest:
                fbest, gbest = t, gn
        print(f"{tag:12s} {M:5d} {N:5d} {K:5d}  {ct:6.0f} {f0:6.0f} {fbest:6.0f} {gbest:3d}  "
              f"{fbest/ct:.3f}  (base {f0/ct:.3f})")
