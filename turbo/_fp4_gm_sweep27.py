"""Round-27: per-shape group_m autotune via the RELIABLE interleaved-A/B protocol
(round-26 proved light-bench autotune buries real wins as noise; interleaved
thermal-matched alternation is the reliable tool). Sweep gm in {1,2,4,8} for ALL
14 shapes, BN/gn from recommend_config. Pure tile->CU permutation = bit-exact.

Per shape x M: compile all gm, run T interleaved trials (each trial times every
gm back-to-back so thermal ramp hits all equally), report per-gm mean + the gm
that wins the most trials vs the gm=4 anchor. A gm!=4 is a REAL win only if it
beats gm4 in the large majority of trials AND mean delta > ~1%."""
import sys, math, torch
_R = "/workspace/code/FlyDSL"
for p in (_R, _R + "/flydsl/src"):
    sys.path.insert(0, p)
import flydsl.compiler as flyc
from turbo.mxfp4_gemm_8wave import compile_mxfp4_gemm_8w, recommend_config
from turbo.mxfp8_gemm_8wave import preshuffle_scale, preshuffle_scale_b_comb
SB = 32
GM_CANDS = [1, 2, 4, 8]
_CC = {}


def get_cc(K, BM, BN, gm, gn):
    key = (K, BM, BN, gm, gn)
    if key not in _CC:
        _CC[key] = compile_mxfp4_gemm_8w(K=K, BLOCK_M=BM, BLOCK_N=BN, mode="pipe", group_m=gm, group_n=gn)
    return _CC[key]


def make(M, N, K):
    d = "cuda"
    a = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device=d)
    b = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=d)
    asc = torch.randint(125, 130, (M, K // SB), dtype=torch.uint8, device=d)
    bsc = torch.randint(125, 130, (N, K // SB), dtype=torch.uint8, device=d)
    asp = preshuffle_scale(asc, K, 4)
    return a.view(torch.int8).view(-1), b.view(torch.int8).view(-1), asp.view(-1), bsc


def t1(cc, ar, it=40):
    e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(); e0.record()
    for _ in range(it):
        cc(*ar)
    e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1) * 1000 / it


SHAPES = [
    ("7B q/o", 4096, 4096), ("7B gate/up", 11008, 4096), ("7B down", 4096, 11008),
    ("70B q/o", 8192, 8192), ("70B kv", 1024, 8192),
    ("70B gate/up", 28672, 8192), ("70B down", 8192, 28672),
]
T = 8
print("=== per-shape group_m interleaved sweep (gm in {1,2,4,8}, anchor gm4) ===", flush=True)
for M in (4096, 8192):
    print(f"--- M={M} ---", flush=True)
    for tag, N, K in SHAPES:
        inp = make(M, N, K)
        ai, bi, asp, bsc = inp
        BM, BN, _, gn = recommend_config(M, N, K)
        bsp = (preshuffle_scale_b_comb(bsc, K) if BN >= 256
               else preshuffle_scale(bsc, K, BN // 128)).view(-1)
        c = torch.zeros((M, N), dtype=torch.bfloat16, device="cuda")
        st = torch.cuda.current_stream()
        ar = (ai, bi, c.view(-1), asp, bsp, M, N, st)
        ccs = {gm: flyc.compile(get_cc(K, BM, BN, gm, gn), *ar) for gm in GM_CANDS}
        for gm in GM_CANDS:
            for _ in range(15):
                ccs[gm](*ar)
        torch.cuda.synchronize()
        times = {gm: [] for gm in GM_CANDS}
        for _ in range(T):
            for gm in GM_CANDS:
                times[gm].append(t1(ccs[gm], ar))
        tf = {gm: 2 * M * N * K / (sum(times[gm]) / T / 1e6) / 1e12 for gm in GM_CANDS}
        anchor = tf[4]
        wins = {gm: sum(1 for i in range(T) if times[gm][i] < times[4][i]) for gm in GM_CANDS}
        best_gm = max(GM_CANDS, key=lambda g: tf[g])
        flags = " ".join(f"gm{g}={tf[g]:.0f}({wins[g]}/{T})" for g in GM_CANDS)
        note = ""
        if best_gm != 4 and (tf[best_gm] / anchor - 1) > 0.01 and wins[best_gm] >= T - 1:
            note = f"  <-- WIN gm{best_gm} +{(tf[best_gm]/anchor-1)*100:.1f}%"
        print(f"{tag:12s} M{M:5d} N{N:6d} K{K:6d} BN{BN} gn{gn:3d}  {flags}  best=gm{best_gm}{note}", flush=True)
print("DONE", flush=True)
