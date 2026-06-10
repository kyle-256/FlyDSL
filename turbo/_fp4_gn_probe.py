"""Round-22 probe: 70B gate/up group_n per-M (recommend_config fixes nb//8=14;
_fp4_vs_aiter sweep picked gn=28 for M4096, gn=14 for M8192). Confirm whether
M4096 gn=28 is a stable signal (not single-run noise) and M8192 stays best at 14.
Paired rotating bench vs aiter; SNR check that band swizzle (gn) is bit-exact
permutation (gn=14 vs gn=28 must produce ~identical output)."""
import sys, os, math, statistics
import torch
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


def bench(fn, it=30, reps=4):
    fn(); torch.cuda.synchronize()
    best = 1e9
    for _ in range(reps):
        e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(); e0.record()
        for _ in range(it):
            fn()
        e1.record(); torch.cuda.synchronize(); best = min(best, e0.elapsed_time(e1) * 1000 / it)
    return best


def aiter_us(M, N, K):
    x = torch.randn((M, K), dtype=torch.bfloat16, device="cuda")
    w = torch.randn((N, K), dtype=torch.bfloat16, device="cuda")
    xq, xs = quant_func(x, shuffle=True); wq, ws = quant_func(w, shuffle=True)
    wsh = shuffle_weight(wq, layout=(16, 16))
    fn = lambda: aiter.gemm_a4w4(xq, wsh, xs, ws, bpreshuffle=True); fn()
    for _ in range(10): fn()
    return bench(fn)


def snr(o, r):
    d = (o.float() - r.float()); s = (r.float() ** 2).sum().item(); n = (d ** 2).sum().item()
    return 99.0 if n == 0 else 10 * math.log10(s / max(n, 1e-30))


def make(M, N, K, gn):
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
    return cc, ar, c


N, K = 28672, 8192
GNS = [14, 28]
ROUNDS = 7
for M in (4096, 8192):
    ai_us = aiter_us(M, N, K)
    ccs = {gn: make(M, N, K, gn) for gn in GNS}
    # correctness: band swizzle is pure permutation -> gn=28 output must match gn=14
    g14 = ccs[14]; g14[2].zero_(); g14[0](*g14[1]); torch.cuda.synchronize(); gold = g14[2].clone()
    g28 = ccs[28]; g28[2].zero_(); g28[0](*g28[1]); torch.cuda.synchronize()
    snr_28 = snr(g28[2], gold)
    for gn in GNS:
        cc, ar, _ = ccs[gn]
        for _ in range(15): cc(*ar)
    torch.cuda.synchronize()
    samp = {gn: [] for gn in GNS}
    for r in range(ROUNDS):
        for gn in GNS:
            cc, ar, _ = ccs[gn]
            us = bench(lambda: cc(*ar), it=30, reps=1)
            samp[gn].append(2 * M * N * K / (us / 1e6) / 1e12)
    print(f"=== 70B gate/up M={M} N={N} K={K}  aiter={2*M*N*K/(ai_us/1e6)/1e12:.0f}TF  SNR(gn28 vs gn14)={snr_28:.1f}dB ===")
    base = statistics.median(samp[14])
    for gn in GNS:
        med = statistics.median(samp[gn])
        ai_tf = 2 * M * N * K / (ai_us / 1e6) / 1e12
        print(f"  gn={gn:3d}  fly_med={med:6.0f}TF  fly/aiter={med/ai_tf:.4f}  vs_gn14={med/base:.4f}")
    sys.stdout.flush()
