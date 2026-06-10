"""round-63: joint (group_m x group_n) reliable-interleaved sweep on 70B gate/up
(N=28672, K=8192) — the LARGEST band-routed big-N shape, whose production config
(group_m=4, group_n=nb//8=14) was DERIVED from the fp8 skill sweet-spot (GM4xGN14),
NEVER joint-swept under the reliable interleaved protocol for THIS shape.

r61 lesson: a per-dimension config sweep can MISS a joint (gm x gn) interaction
optimum; joint-interleaved surfaced the big-K down win that r28's 1-D sweep missed.
Apply that same protocol to the one big-N shape it never covered. big-N is the band
lever's main battleground (L2-reuse), 70B gate/up is at 0.778 fly/aiter (headroom),
and it's a real Llama layer (K1, transfers 1:1).

ANCHOR = production (gm4, gn14). Candidates = other (gm, gn) cells. Pure tile->CU
permutation => bit-exact / det-neutral; only perf in question. A reliable >=1%
winner (>=7/10 pairs AND cross-M consistent) routes per-shape into recommend_config
(ACCEPT); else ROLLBACK (the 1-D-derived gm4/gn14 IS the joint optimum, dim closed).
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
    bsp = preshuffle_scale_b_comb(bsc, K).view(-1)
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
SHAPES = [("70Bgateup", 28672, 8192)]
for tag, N, K in SHAPES:
    for M in (4096, 8192):
        nb = N // 256
        gn0 = nb // 8  # production anchor group_n = 14
        a, b, asc, bsc = mkraw(M, N, K)
        tf = lambda ms: 2 * M * N * K / (ms / 1e6) / 1e12
        ac, aar, abase = build(M, N, K, a, b, asc, bsc, 4, gn0)  # production anchor (gm4, gn14)
        base = abase.clone()
        # joint cartesian around the production point: vary gm at gn14, vary gn at gm4, + cross
        cands = [(1, gn0), (2, gn0), (8, gn0), (16, gn0),
                 (4, nb // 16), (4, nb // 4), (4, nb // 2),
                 (8, nb // 16), (8, nb // 4), (2, nb // 4), (16, nb // 16)]
        seen = set((4, gn0)); cands = [c for c in cands if not (c in seen or seen.add(c))]
        print(f"=== {tag} M={M} N={N} K={K} nb={nb}  anchor=(gm4,gn{gn0})={tf(bench(ac, aar)):.0f}TF ===", flush=True)
        for (gm, gn) in [(8, gn0), (4, nb // 4)]:
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
