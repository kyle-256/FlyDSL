"""Consolidated dense mxfp4: baseline (BM256/gn0) vs recommend_config() across
all Llama 7B/70B shapes, M in {4096,8192}. Verifies bit-exact-ish (SNR) + det
(two runs identical) + reports net TFLOPS gain."""
import sys, os, torch
_R = "/workspace/code/FlyDSL"
for p in (_R, _R + "/flydsl/src"):
    sys.path.insert(0, p)
import flydsl.compiler as flyc
from turbo.mxfp4_gemm_8wave import compile_mxfp4_gemm_8w, recommend_config
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
    asp = preshuffle_scale(asc, K, 4)
    return a.view(torch.int8).view(-1), b.view(torch.int8).view(-1), asp.view(-1), bsc


def one(M, N, K, BM, BN, gm, gn, inp):
    ai, bi, asp, bsc = inp
    # B-scale format follows the chosen BLOCK_N: comb (4-scale dwordx4) for BN256,
    # per-region (preshuffle_scale ..., BN//128) for BN128.
    bsp = (preshuffle_scale_b_comb(bsc, K) if BN >= 256
           else preshuffle_scale(bsc, K, BN // 128)).view(-1)
    c = torch.zeros((M, N), dtype=torch.bfloat16, device="cuda")
    st = torch.cuda.current_stream()
    fn = compile_mxfp4_gemm_8w(K=K, BLOCK_M=BM, BLOCK_N=BN, mode="pipe", group_m=gm, group_n=gn)
    ar = (ai, bi, c.view(-1), asp, bsp, M, N, st)
    cc = flyc.compile(fn, *ar); cc(*ar); torch.cuda.synchronize()
    out1 = c.clone()
    c.zero_(); cc(*ar); torch.cuda.synchronize(); out2 = c.clone()  # det check
    det = torch.equal(out1, out2)
    us = bench(lambda: cc(*ar))
    return 2 * M * N * K / (us / 1e6) / 1e12, out1, det


SHAPES = [
    ("7B q/o", 4096, 4096), ("7B gate/up", 11008, 4096), ("7B down", 4096, 11008),
    ("70B q/o", 8192, 8192), ("70B kv", 1024, 8192),
    ("70B gate/up", 28672, 8192), ("70B down", 8192, 28672),
]
print(f"{'shape':12s} {'M':>5s} {'N':>5s} {'K':>5s}  {'base':>5s} {'reco':>5s} {'BM':>3s} {'BN':>3s} {'gm':>2s} {'gn':>3s}  {'win%':>5s}  SNR   det")
import math as _m
lr_b = lr_r = 0.0
worst_win = 1e9
for M in (4096, 8192):
    print(f"--- M={M} ---")
    for tag, N, K in SHAPES:
        inp = make(M, N, K)
        BM, BN, gm, gn = recommend_config(M, N, K)
        tfb, refb, detb = one(M, N, K, 256, 256, 4, 0, inp)
        tfr, outr, detr = one(M, N, K, BM, BN, gm, gn, inp)
        s = snr(outr, refb)
        win = (tfr / tfb - 1) * 100
        worst_win = min(worst_win, win)
        lr_b += _m.log(tfb); lr_r += _m.log(tfr)
        print(f"{tag:12s} {M:5d} {N:5d} {K:5d}  {tfb:5.0f} {tfr:5.0f} {BM:3d} {BN:3d} {gm:2d} {gn:3d}  {win:+5.1f}  {s:5.0f}  {'OK' if detr else 'NO'}")
gb = _m.exp(lr_b / 14); gr = _m.exp(lr_r / 14)
print(f"GEOMEAN TFLOPS base {gb:.0f} -> reco {gr:.0f}  ({(gr/gb-1)*100:+.2f}%); worst per-shape win {worst_win:+.1f}%")
