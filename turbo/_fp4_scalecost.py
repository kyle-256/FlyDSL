"""DIAGNOSTIC: how much does the scale-load path cost? Compare production pipe
(full per-1x32 E8M0 scale load + scaled MFMA) vs pipe const_scale (identity
0x7F7F7F7F immediate, NO scale buffer_loads). const_scale output is numerically
wrong -- this is a PERF-only probe to localize the bulk gap to the scale path.
"""
import sys, math, statistics
import torch
_R = "/workspace/code/FlyDSL"
for p in (_R, _R + "/flydsl/src"):
    sys.path.insert(0, p)
import flydsl.compiler as flyc
from turbo.mxfp4_gemm_8wave import compile_mxfp4_gemm_8w, recommend_config
from turbo.mxfp8_gemm_8wave import preshuffle_scale, preshuffle_scale_b_comb

SB = 32


def make_inputs(M, N, K):
    d = "cuda"
    a = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device=d)
    b = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=d)
    asc = torch.randint(125, 130, (M, K // SB), dtype=torch.uint8, device=d)
    bsc = torch.randint(125, 130, (N, K // SB), dtype=torch.uint8, device=d)
    c = torch.zeros((M, N), dtype=torch.bfloat16, device=d)
    asp = preshuffle_scale(asc, K, 4)
    bsp = preshuffle_scale_b_comb(bsc, K)
    return (a.view(torch.int8).view(-1), b.view(torch.int8).view(-1), c.view(-1),
            asp.view(-1), bsp.view(-1), M, N, torch.cuda.current_stream()), c


def meas(cc, ar, it=40):
    e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(); e0.record()
    for _ in range(it): cc(*ar)
    e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1) * 1000 / it


SHAPES = [
    ("7Bqo",   4096, 4096, 4096),
    ("7Bdn",   4096, 4096, 11008),
    ("70Bqo",  4096, 8192, 8192),
    ("70Bdn",  4096, 8192, 28672),
]
VARIANTS = [("pipe", dict(mode="pipe")), ("pipe_noscale", dict(mode="pipe", const_scale=True))]
ROUNDS = 7

for tag, M, N, K in SHAPES:
    BM, BN, gm, gn, nx = recommend_config(M, N, K)
    if BN < 256:
        BN = 256; gm = 4
    ar, c = make_inputs(M, N, K)
    ccs = {}
    for vn, kw in VARIANTS:
        fn = compile_mxfp4_gemm_8w(K=K, BLOCK_M=BM, BLOCK_N=BN, group_m=gm, group_n=gn, num_xcds=nx, **kw)
        cc = flyc.compile(fn, *ar); cc(*ar); torch.cuda.synchronize()
        ccs[vn] = cc
    for vn in ccs:
        for _ in range(20): ccs[vn](*ar)
    torch.cuda.synchronize()
    samp = {vn: [] for vn in ccs}
    for r in range(ROUNDS):
        for vn in ccs:
            samp[vn].append(meas(ccs[vn], ar))
    base = statistics.median(samp["pipe"])
    print(f"=== {tag} M{M} N{N} K{K} BM{BM} BN{BN} ===")
    for vn in ccs:
        med = statistics.median(samp[vn]); tf = 2 * M * N * K / (med / 1e6) / 1e12
        print(f"  {vn:13s} med={med:8.2f}us tf={tf:6.0f} speedup_vs_pipe={base/med:.4f}")
    sys.stdout.flush()
