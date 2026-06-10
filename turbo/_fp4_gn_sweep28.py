"""Round-28: per-shape group_n (band swizzle) sweep via the RELIABLE interleaved-A/B
protocol. round-26 judged mid-N band noise via light-bench + best-of-4; round-27
proved interleaved is more reliable. band is a verified lever (big-N +8%); mid-N
(N=8192) has a non-zero L2-reuse-deficit. Sweep gn on mid/big-N BN256 shapes,
gm fixed at reco (4). Pure tile->CU permutation = bit-exact. Closes the group_n
lever definitively (last cheap Mode-A config knob).

Anchor = current production gn (0 for mid-N, 14 for 70B gate/up). A gn != anchor
is a REAL win only if it beats anchor in the large majority of T trials AND mean
delta > ~1%."""
import sys, torch
_R = "/workspace/code/FlyDSL"
for p in (_R, _R + "/flydsl/src"):
    sys.path.insert(0, p)
import flydsl.compiler as flyc
from turbo.mxfp4_gemm_8wave import compile_mxfp4_gemm_8w
from turbo.mxfp8_gemm_8wave import preshuffle_scale, preshuffle_scale_b_comb
SB = 32
_CC = {}


def get_cc(K, BN, gm, gn):
    key = (K, BN, gm, gn)
    if key not in _CC:
        _CC[key] = compile_mxfp4_gemm_8w(K=K, BLOCK_M=256, BLOCK_N=BN, mode="pipe", group_m=gm, group_n=gn)
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


# (tag, N, K, anchor_gn, [gn candidates])
SHAPES = [
    ("70B q/o", 8192, 8192, 0, [0, 4, 8]),
    ("70B down", 8192, 28672, 0, [0, 4, 8]),
    ("70B gate/up", 28672, 8192, 14, [14, 28, 56]),
]
T = 8
print("=== per-shape group_n band interleaved sweep (BN256, gm4) ===", flush=True)
for M in (4096, 8192):
    print(f"--- M={M} ---", flush=True)
    for tag, N, K, anchor, gns in SHAPES:
        inp = make(M, N, K)
        ai, bi, asp, bsc = inp
        bsp = preshuffle_scale_b_comb(bsc, K).view(-1)
        c = torch.zeros((M, N), dtype=torch.bfloat16, device="cuda")
        st = torch.cuda.current_stream()
        ar = (ai, bi, c.view(-1), asp, bsp, M, N, st)
        ccs = {gn: flyc.compile(get_cc(K, 256, 4, gn), *ar) for gn in gns}
        for gn in gns:
            for _ in range(15):
                ccs[gn](*ar)
        torch.cuda.synchronize()
        times = {gn: [] for gn in gns}
        for _ in range(T):
            for gn in gns:
                times[gn].append(t1(ccs[gn], ar))
        tf = {gn: 2 * M * N * K / (sum(times[gn]) / T / 1e6) / 1e12 for gn in gns}
        a_tf = tf[anchor]
        wins = {gn: sum(1 for i in range(T) if times[gn][i] < times[anchor][i]) for gn in gns}
        best = max(gns, key=lambda g: tf[g])
        flags = " ".join(f"gn{g}={tf[g]:.0f}({wins[g]}/{T})" for g in gns)
        note = ""
        if best != anchor and (tf[best] / a_tf - 1) > 0.01 and wins[best] >= T - 1:
            note = f"  <-- WIN gn{best} +{(tf[best]/a_tf-1)*100:.1f}%"
        print(f"{tag:12s} M{M:5d} N{N:6d} K{K:6d} anchor=gn{anchor}  {flags}  best=gn{best}{note}", flush=True)
print("DONE", flush=True)
