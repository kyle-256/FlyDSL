"""round-55: re-verify group_m for the NEW BM128 kv config (r54 changed kv M4096
BM192->BM128). The landed gm2 (r26) was tuned for the BM192 grid (22 M-tiles x 8
N-tiles = 176 tiles); BM128 is a different grid (32 M-tiles x 8 = 256 tiles), so
the optimal GROUP_M super-block clustering for XCD/L2 locality may differ. This is
the natural follow-up to r54's config change (re-verify the kv-specific group_m,
not a blind re-sweep of the noise-confirmed general group_m).

Pure tile->CU permutation = bit-exact / det-neutral; the only question is perf.
Interleaved A/B (thermal-matched) + win-count vs the gm2 anchor, kv M4096 BM128
(binding min_ratio shape). A reliable >=1% winner (>=7/10 pairs) routes; else
ROLLBACK (gm2 stays, dimension closed for BM128).
"""
import sys, torch, statistics as st_
_R = "/workspace/code/FlyDSL"
for p in (_R, _R + "/flydsl/src"):
    sys.path.insert(0, p)
import flydsl.compiler as flyc
from turbo.mxfp4_gemm_8wave import compile_mxfp4_gemm_8w
from turbo.mxfp8_gemm_8wave import preshuffle_scale
SB = 32


def snr(o, r):
    o = o.float(); r = r.float()
    return 10 * torch.log10((r ** 2).sum() / ((o - r) ** 2).sum().clamp_min(1e-20)).item()


def mkraw(M, N, K):
    d = "cuda"
    a = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device=d)
    b = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=d)
    asc = torch.randint(125, 130, (M, K // SB), dtype=torch.uint8, device=d)
    bsc = torch.randint(125, 130, (N, K // SB), dtype=torch.uint8, device=d)
    return a.view(torch.int8).view(-1), b.view(torch.int8).view(-1), asc, bsc


def ascale(asc, M, K, BM):
    nta = BM // 64
    q = 16 * nta
    pad = ((M + q - 1) // q) * q
    if pad != M:
        ap = torch.zeros((pad, K // SB), dtype=torch.uint8, device=asc.device)
        ap[:M] = asc
        asc = ap
    return preshuffle_scale(asc, K, nta).view(-1)


def build(M, N, K, BM, BN, a, b, asc, bsc, gm):
    asp = ascale(asc, M, K, BM)
    bsp = preshuffle_scale(bsc, K, BN // 128).view(-1)
    c = torch.zeros((M, N), dtype=torch.bfloat16, device="cuda")
    st = torch.cuda.current_stream()
    fn = compile_mxfp4_gemm_8w(K=K, BLOCK_M=BM, BLOCK_N=BN, mode="pipe", group_m=gm, group_n=0)
    ar = (a, b, c.view(-1), asp, bsp, M, N, st)
    cc = flyc.compile(fn, *ar)
    cc(*ar); torch.cuda.synchronize()
    return cc, ar, c


def bench(cc, ar, it=30, wu=10, reps=2):
    for _ in range(wu):
        cc(*ar)
    torch.cuda.synchronize(); best = 1e9
    for _ in range(reps):
        e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(); e0.record()
        for _ in range(it):
            cc(*ar)
        e1.record(); torch.cuda.synchronize(); best = min(best, e0.elapsed_time(e1) * 1000 / it)
    return best


M, N, K = 4096, 1024, 8192
BM, BN = 128, 128
T = 10
a, b, asc, bsc = mkraw(M, N, K)


def tf(ms):
    return 2 * M * N * K / (ms / 1e6) / 1e12


# correctness/det sanity: gm permutation must be bit-exact across gm values
cc2, ar2, c2 = build(M, N, K, BM, BN, a, b, asc, bsc, 2)
base = c2.clone()
for gm in (1, 4, 8):
    ccx, arx, cx = build(M, N, K, BM, BN, a, b, asc, bsc, gm)
    md = (cx.float() - base.float()).abs().max().item()
    print(f"gm={gm} vs gm2 bit-diff maxdiff={md:.4g} (perm => expect 0)", flush=True)

print(f"--- kv M={M} N={N} K={K} BM128/BN128 group_m sweep (interleaved vs gm2) ---", flush=True)
anchor_cc, anchor_ar, _ = build(M, N, K, BM, BN, a, b, asc, bsc, 2)
for gm in (1, 4, 8):
    cand_cc, cand_ar, _ = build(M, N, K, BM, BN, a, b, asc, bsc, gm)
    wins = 0; ta = []; tc = []
    for t in range(T):
        t_anchor = bench(anchor_cc, anchor_ar)
        t_cand = bench(cand_cc, cand_ar)
        ta.append(t_anchor); tc.append(t_cand)
        if t_cand < t_anchor:
            wins += 1
    ma = st_.median(ta); mc = st_.median(tc)
    print(f"gm={gm}: {tf(mc):6.0f}TF vs gm2 {tf(ma):6.0f}TF  gm{gm}/gm2={tf(mc)/tf(ma):.3f}  "
          f"gm{gm} wins {wins}/{T}", flush=True)
print("DONE", flush=True)
