"""Round-26 (campaign mxfp4 dense): per-shape AUTOTUNE lever.

Current production picks config via static heuristic recommend_config() (group_m
fixed=4; group_n = nb//8 only for nb>=96, else 0). This was never a real autotune.

This round measures a REAL per-shape autotune that sweeps
  group_n in {0, nb//8, nb//4, nb//2}  x  group_m in {1,4,8,16}
(BLOCK_M=256 fixed; BLOCK_N from recommend_config, kv->128 else 256), picks the
fastest per (M,N,K), and compares geomean vs the static recommend_config across
all 14 Llama shapes. All candidates are pure tile->CU permutation => bit-exact
(zero correctness risk); we still gate SNR + det on the chosen config.

Accept iff autotune geomean beats recommend_config geomean by > noise.

Staged coordinate-descent (bounds #compiles): stage A sweeps gn at gm=4, stage B
sweeps gm at the gn-winner. Compiled kernels cached in-process keyed on
(K,BM,BN,gm,gn) [M-independent]. Hot best-of-N event timing.
"""
import sys, os, math, torch
_R = "/workspace/code/FlyDSL"
for p in (_R, _R + "/flydsl/src"):
    sys.path.insert(0, p)
import flydsl.compiler as flyc
from turbo.mxfp4_gemm_8wave import compile_mxfp4_gemm_8w, recommend_config
from turbo.mxfp8_gemm_8wave import preshuffle_scale, preshuffle_scale_b_comb
SB = 32

_CC = {}  # (K,BM,BN,gm,gn) -> compiled callable (M-independent kernel)


def get_cc(K, BM, BN, gm, gn):
    key = (K, BM, BN, gm, gn)
    cc = _CC.get(key)
    if cc is None:
        fn = compile_mxfp4_gemm_8w(K=K, BLOCK_M=BM, BLOCK_N=BN, mode="pipe", group_m=gm, group_n=gn)
        _CC[key] = cc = fn
    return cc


def bench(fn, it, wu, reps):
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


def run_cfg(M, N, K, BM, BN, gm, gn, inp, bsp, it, wu, reps):
    ai, bi, asp, _ = inp
    c = torch.zeros((M, N), dtype=torch.bfloat16, device="cuda")
    st = torch.cuda.current_stream()
    fn = get_cc(K, BM, BN, gm, gn)
    ar = (ai, bi, c.view(-1), asp, bsp, M, N, st)
    cc = flyc.compile(fn, *ar)
    cc(*ar); torch.cuda.synchronize()
    out1 = c.clone()
    c.zero_(); cc(*ar); torch.cuda.synchronize(); det = torch.equal(out1, c)
    us = bench(lambda: cc(*ar), it, wu, reps)
    return 2 * M * N * K / (us / 1e6) / 1e12, out1, det


def gn_cands(nb):
    return sorted({g for g in (0, nb // 8, nb // 4, nb // 2) if g == 0 or 1 <= g <= nb})


SHAPES = [
    ("7B q/o", 4096, 4096), ("7B gate/up", 11008, 4096), ("7B down", 4096, 11008),
    ("70B q/o", 8192, 8192), ("70B kv", 1024, 8192),
    ("70B gate/up", 28672, 8192), ("70B down", 8192, 28672),
]
GM_CANDS = [1, 4, 8, 16]

print("=== AUTOTUNE SELECTION (staged coordinate descent, light bench it=10 wu=3 reps=2) ===", flush=True)
chosen = {}  # (M,N,K) -> (BM,BN,gm,gn)
for M in (4096, 8192):
    for tag, N, K in SHAPES:
        inp = make(M, N, K)
        BM, BN, reco_gn = recommend_config(M, N, K)
        bsp = (preshuffle_scale_b_comb(inp[3], K) if BN >= 256
               else preshuffle_scale(inp[3], K, BN // 128)).view(-1)
        nb = N // BN
        # stage A: gn sweep at gm=4
        best = None
        for gn in gn_cands(nb):
            tf, _, det = run_cfg(M, N, K, BM, BN, 4, gn, inp, bsp, 10, 3, 2)
            if best is None or tf > best[0]:
                best = (tf, 4, gn)
            print(f"  [{tag} M{M}] gm=4 gn={gn:3d} -> {tf:5.0f} TF det={'OK' if det else 'NO'}", flush=True)
        # stage B: gm sweep at best gn
        best_gn = best[2]
        for gm in GM_CANDS:
            if gm == 4:
                continue
            tf, _, det = run_cfg(M, N, K, BM, BN, gm, best_gn, inp, bsp, 10, 3, 2)
            if tf > best[0]:
                best = (tf, gm, best_gn)
            print(f"  [{tag} M{M}] gm={gm:2d} gn={best_gn:3d} -> {tf:5.0f} TF det={'OK' if det else 'NO'}", flush=True)
        chosen[(M, N, K)] = (BM, BN, best[1], best[2])
        print(f"  [{tag} M{M}] CHOSEN gm={best[1]} gn={best[2]} (reco gm=4 gn={reco_gn})", flush=True)

print("\n=== CLEAN CONFIRM (canonical bench it=30 wu=8 reps=4): base vs reco vs auto ===", flush=True)
print(f"{'shape':12s} {'M':>5s} {'N':>6s} {'K':>6s}  {'base':>5s} {'reco':>5s} {'auto':>5s}  "
      f"{'r/b%':>5s} {'a/b%':>5s} {'a/r%':>5s}  {'gm':>2s} {'gn':>3s}  SNR det", flush=True)
import math as _m
lr_b = lr_r = lr_a = 0.0  # log-sums for geomean
worst_ar = 1e9
for M in (4096, 8192):
    print(f"--- M={M} ---", flush=True)
    for tag, N, K in SHAPES:
        inp = make(M, N, K)
        BM, BN, reco_gn = recommend_config(M, N, K)
        bsp = (preshuffle_scale_b_comb(inp[3], K) if BN >= 256
               else preshuffle_scale(inp[3], K, BN // 128)).view(-1)
        _, _, agm, agn = chosen[(M, N, K)]
        tfb, refb, _ = run_cfg(M, N, K, 256, 256, 4, 0, inp, preshuffle_scale_b_comb(inp[3], K).view(-1), 30, 8, 4)
        tfr, _, _ = run_cfg(M, N, K, BM, BN, 4, reco_gn, inp, bsp, 30, 8, 4)
        tfa, outa, deta = run_cfg(M, N, K, BM, BN, agm, agn, inp, bsp, 30, 8, 4)
        s = snr(outa, refb)
        rb = (tfr / tfb - 1) * 100; ab = (tfa / tfb - 1) * 100; ar_ = (tfa / tfr - 1) * 100
        worst_ar = min(worst_ar, ar_)
        lr_b += _m.log(tfb); lr_r += _m.log(tfr); lr_a += _m.log(tfa)
        print(f"{tag:12s} {M:5d} {N:6d} {K:6d}  {tfb:5.0f} {tfr:5.0f} {tfa:5.0f}  "
              f"{rb:+5.1f} {ab:+5.1f} {ar_:+5.1f}  {agm:2d} {agn:3d}  {s:4.0f} {'OK' if deta else 'NO'}", flush=True)
n = 14
gb = _m.exp(lr_b / n); gr = _m.exp(lr_r / n); ga = _m.exp(lr_a / n)
print(f"\nGEOMEAN TFLOPS: base {gb:.0f}  reco {gr:.0f}  auto {ga:.0f}", flush=True)
print(f"reco/base {(gr/gb-1)*100:+.2f}%  auto/base {(ga/gb-1)*100:+.2f}%  auto/reco {(ga/gr-1)*100:+.2f}%  "
      f"worst per-shape auto/reco {worst_ar:+.1f}%", flush=True)
print("DONE", flush=True)
