"""Round-26 diagnostic: is kv (N=1024) gm=1 a robust lever vs reco gm=4?
3-trial canonical bench per config, both M. Noise-assessment protocol."""
import sys, os, math, torch
_R = "/workspace/code/FlyDSL"
for p in (_R, _R + "/flydsl/src"):
    sys.path.insert(0, p)
import flydsl.compiler as flyc
from turbo.mxfp4_gemm_8wave import compile_mxfp4_gemm_8w
from turbo.mxfp8_gemm_8wave import preshuffle_scale
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


def make(M, N, K):
    d = "cuda"
    a = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device=d)
    b = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=d)
    asc = torch.randint(125, 130, (M, K // SB), dtype=torch.uint8, device=d)
    bsc = torch.randint(125, 130, (N, K // SB), dtype=torch.uint8, device=d)
    asp = preshuffle_scale(asc, K, 4)
    return a.view(torch.int8).view(-1), b.view(torch.int8).view(-1), asp.view(-1), bsc


N, K = 1024, 8192
BN = 128
CC = {}
for gm in (1, 2, 4):
    CC[gm] = compile_mxfp4_gemm_8w(K=K, BLOCK_M=256, BLOCK_N=BN, mode="pipe", group_m=gm, group_n=0)

for M in (4096, 8192):
    inp = make(M, N, K)
    ai, bi, asp, bsc = inp
    bsp = preshuffle_scale(bsc, K, BN // 128).view(-1)
    st = torch.cuda.current_stream()
    print(f"--- kv M={M} N={N} K={K} (BN=128) ---", flush=True)
    for gm in (4, 1, 2):
        c = torch.zeros((M, N), dtype=torch.bfloat16, device="cuda")
        ar = (ai, bi, c.view(-1), asp, bsp, M, N, st)
        cc = flyc.compile(CC[gm], *ar)
        tfs = []
        for t in range(3):
            us = bench(lambda: cc(*ar))
            tfs.append(2 * M * N * K / (us / 1e6) / 1e12)
        mean = sum(tfs) / len(tfs)
        print(f"  gm={gm}: trials {[f'{x:.0f}' for x in tfs]}  mean {mean:.0f} TF", flush=True)
print("DONE", flush=True)
