"""round-61: JOINT (group_m x group_n) reliable-interleaved autotune for the BULK
regime (the geomean driver, 8/14 shapes @ 0.70-0.81). r26 ran a gm x gn autotune
but NON-interleaved (full-bench, unreliable per the campaign's own noise lesson);
r27 swept gm alone (fixed gn) and r28 swept gn alone (fixed gm=4) -> both judged
noise/hurt. NEITHER tested the JOINT cartesian product under the reliable
interleaved A/B protocol. A per-shape optimum could need gm!=4 AND gn!=0 together
(an L2-locality interaction that a 1-D sweep at the wrong fixed other-axis misses).

Pure tile->CU permutation = bit-exact / det-neutral; only perf is in question.
For each bulk shape: interleaved A/B (thermal-matched, shared inputs) + win-count
vs the production anchor (gm=4, gn=0). A reliable >=1% winner (>=7/10 pairs AND
cross-M consistent) would route into recommend_config (ACCEPT); else ROLLBACK
(joint = noise too, closes the gm x gn interaction question for bulk).
"""
import sys, torch, statistics as st_
_R = "/workspace/code/FlyDSL"
for p in (_R, _R + "/flydsl/src"):
    sys.path.insert(0, p)
import flydsl.compiler as flyc
from turbo.mxfp4_gemm_8wave import compile_mxfp4_gemm_8w
from turbo.mxfp8_gemm_8wave import preshuffle_scale, preshuffle_scale_b_comb
SB = 32


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


def build(M, N, K, a, b, asc, bsc, gm, gn, BM=256, BN=256):
    asp = ascale(asc, M, K, BM)
    bsp = preshuffle_scale_b_comb(bsc, K).view(-1)  # BN=256 combined B-scale
    c = torch.zeros((M, N), dtype=torch.bfloat16, device="cuda")
    st = torch.cuda.current_stream()
    fn = compile_mxfp4_gemm_8w(K=K, BLOCK_M=BM, BLOCK_N=BN, mode="pipe", group_m=gm, group_n=gn)
    ar = (a, b, c.view(-1), asp, bsp, M, N, st)
    cc = flyc.compile(fn, *ar)
    cc(*ar); torch.cuda.synchronize()
    return cc, ar, c


def bench(cc, ar, it=30, wu=8, reps=2):
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


T = 10
# bulk geomean-driver shapes: 70B q/o (square, L2 mid) + 70B down (big-K, L2 ~66%, r28-hurt control)
SHAPES = [("70Bqo", 8192, 8192), ("70Bdown", 8192, 28672)]
for tag, N, K in SHAPES:
    for M in (4096, 8192):
        nb = N // 256
        a, b, asc, bsc = mkraw(M, N, K)
        tf = lambda ms: 2 * M * N * K / (ms / 1e6) / 1e12
        # bit-exact check: joint candidate must equal anchor (pure permutation)
        ac, aar, abase = build(M, N, K, a, b, asc, bsc, 4, 0)
        base = abase.clone()
        # joint candidate grid: the NEW cross cells (gm!=4 x gn!=0) + 1-D controls
        cands = [(1, 0), (8, 0), (16, 0),
                 (4, nb // 8), (8, nb // 8), (16, nb // 8), (1, nb // 8),
                 (8, nb // 4), (4, nb // 4)]
        print(f"=== {tag} M={M} N={N} K={K} nb={nb}  anchor=(gm4,gn0)={tf(bench(ac, aar)):.0f}TF ===", flush=True)
        # sanity bit-exact on a couple of cross cells
        for (gm, gn) in [(8, nb // 8), (16, 0)]:
            cx, arx, cc_ = build(M, N, K, a, b, asc, bsc, gm, gn)
            md = (cc_.float() - base.float()).abs().max().item()
            print(f"  bit-diff (gm{gm},gn{gn}) vs anchor maxdiff={md:.4g} (perm=>0)", flush=True)
        for (gm, gn) in cands:
            cc, car, _ = build(M, N, K, a, b, asc, bsc, gm, gn)
            wins = 0; ta = []; tc = []
            for t in range(T):
                t_a = bench(ac, aar); t_c = bench(cc, car)
                ta.append(t_a); tc.append(t_c)
                wins += (t_c < t_a)
            ma = st_.median(ta); mc = st_.median(tc)
            flag = "  <-- WIN?" if (mc < ma * 0.99 and wins >= 7) else ""
            print(f"  (gm{gm:2d},gn{gn:2d}): {tf(mc):6.0f}TF vs {tf(ma):6.0f}TF  ratio={tf(mc)/tf(ma):.3f}  wins {wins}/{T}{flag}", flush=True)
print("DONE", flush=True)
