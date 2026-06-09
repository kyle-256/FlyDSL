"""r_k5 finalize: pipe-BN128 (combined-G2S, conservative wait_barrier(0)) det0 + perf
vs pipe-BN256 baseline on kv. Expectation: conservative waits ~= staged perf (loses);
r_k6 tightens the main-loop wait to recover pipelining."""
import os, sys, torch
_R = "/workspace/code/FlyDSL"
for p in (_R, _R + "/flydsl/src"):
    sys.path.insert(0, p)
import flydsl.compiler as flyc
from turbo.mxfp4_gemm_8wave import compile_mxfp4_gemm_8w
from turbo.mxfp8_gemm_8wave import preshuffle_scale, preshuffle_scale_b_comb
SB = 32


def bench(fn, it=30, wu=10, reps=3):
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


def mk(M, N, K):
    d = "cuda"
    return (torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device=d),
            torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=d),
            torch.randint(125, 130, (M, K // SB), dtype=torch.uint8, device=d),
            torch.randint(125, 130, (N, K // SB), dtype=torch.uint8, device=d))


def cfg(M, N, K, BN, a, b, asc, bsc):
    d = "cuda"
    asp = preshuffle_scale(asc, K, 256 // 64)
    bsp = preshuffle_scale_b_comb(bsc, K) if BN >= 256 else preshuffle_scale(bsc, K, BN // 128)
    c = torch.zeros((M, N), dtype=torch.bfloat16, device=d)
    st = torch.cuda.current_stream()
    fn = compile_mxfp4_gemm_8w(K=K, BLOCK_M=256, BLOCK_N=BN, mode="pipe")
    args = (a.view(torch.int8).view(-1), b.view(torch.int8).view(-1), c.view(-1),
            asp.view(-1), bsp.view(-1), M, N, st)
    cc = flyc.compile(fn, *args)
    tf = 2 * M * N * K / (bench(lambda: cc(*args)) / 1e6) / 1e12
    return tf, cc, args, c


def det(cc, args, c, runs=300):
    cc(*args); torch.cuda.synchronize(); ref = c.clone(); md = 0.0
    for _ in range(runs):
        c.zero_(); cc(*args); torch.cuda.synchronize()
        md = max(md, (c.float() - ref.float()).abs().max().item())
    return md


for M, N, K in [(4096, 1024, 8192), (8192, 1024, 8192)]:
    a, b, asc, bsc = mk(M, N, K)
    t256, *_ = cfg(M, N, K, 256, a, b, asc, bsc)
    t128, cc8, ar8, c8 = cfg(M, N, K, 128, a, b, asc, bsc)
    md = det(cc8, ar8, c8, 300)
    print(f"kv M={M} N={N} K={K}: pipe-BN256={t256:6.0f}TF | pipe-BN128(cons-wait)={t128:6.0f}TF "
          f"BN128/256={t128/t256:.3f} | det(300)={md:.6g} {'DET0' if md==0 else 'NONDET'}")
