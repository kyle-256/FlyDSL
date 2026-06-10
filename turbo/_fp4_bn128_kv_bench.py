"""r_k4 gating: does staged-BLOCK_N=128 (full grid) beat pipe-BLOCK_N=256 (1/4 grid)
on kv (N=1024, occupancy-bound)?  Separates the two effects:
  pipe-BN256   = current production baseline
  staged-BN256 = staged pipelining-loss control (same grid as pipe)
  staged-BN128 = candidate (2x N-tiles -> full grid, but staged has no SW pipeline)
hot best-of-3. Also a det check on staged-BN128 (>=300 same-input runs, bitwise)."""
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
    a = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device=d)
    b = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=d)
    asc = torch.randint(125, 130, (M, K // SB), dtype=torch.uint8, device=d)
    bsc = torch.randint(125, 130, (N, K // SB), dtype=torch.uint8, device=d)
    return a, b, asc, bsc


def cfg(M, N, K, mode, BN, a, b, asc, bsc):
    d = "cuda"
    asp = preshuffle_scale(asc, K, 256 // 64)
    bsp = preshuffle_scale_b_comb(bsc, K) if BN >= 256 else preshuffle_scale(bsc, K, BN // 128)
    c = torch.zeros((M, N), dtype=torch.bfloat16, device=d)
    st = torch.cuda.current_stream()
    fn = compile_mxfp4_gemm_8w(K=K, BLOCK_M=256, BLOCK_N=BN, mode=mode)
    args = (a.view(torch.int8).view(-1), b.view(torch.int8).view(-1), c.view(-1),
            asp.view(-1), bsp.view(-1), M, N, st)
    cc = flyc.compile(fn, *args)
    call = lambda: cc(*args)
    us = bench(call)
    tf = 2 * M * N * K / (us / 1e6) / 1e12
    return tf, us, cc, args, c


def det_check(cc, args, c, runs=300):
    cc(*args); torch.cuda.synchronize()
    ref = c.clone()
    maxd = 0.0
    for _ in range(runs):
        c.zero_(); cc(*args); torch.cuda.synchronize()
        maxd = max(maxd, (c.float() - ref.float()).abs().max().item())
    return maxd


for M, N, K in [(4096, 1024, 8192), (8192, 1024, 8192)]:
    a, b, asc, bsc = mk(M, N, K)
    tp, up, *_ = cfg(M, N, K, "pipe", 256, a, b, asc, bsc)
    ts, us, *_ = cfg(M, N, K, "staged", 256, a, b, asc, bsc)
    t8, u8, cc8, ar8, c8 = cfg(M, N, K, "staged", 128, a, b, asc, bsc)
    print(f"kv M={M} N={N} K={K}:")
    print(f"   pipe   BN256 = {tp:6.0f} TF ({up:.1f}us)  [baseline]")
    print(f"   staged BN256 = {ts:6.0f} TF ({us:.1f}us)  [staged-loss control]  staged/pipe={ts/tp:.3f}")
    print(f"   staged BN128 = {t8:6.0f} TF ({u8:.1f}us)  [candidate]            BN128/pipe={t8/tp:.3f}")
    md = det_check(cc8, ar8, c8, runs=300)
    print(f"   staged BN128 det (300-run same-input): maxdiff={md:.6g}  {'DET0' if md==0 else 'NONDET'}")
