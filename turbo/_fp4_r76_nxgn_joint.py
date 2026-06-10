"""round-76: joint (num_xcds x group_n) ALIGNED-diagonal reliable-interleaved sweep
on the band-routed shapes (70B gate/up big-N + 70B down / 7B down big-K).

r61 lesson: a per-DIMENSION config sweep can MISS a joint interaction optimum.
num_xcds was only ever swept 1-D: r38/r56 fixed group_n (and r56 was kv-only);
r63 swept group_n with num_xcds=8 FIXED. The ALIGNED diagonal
  #bands == num_xcds  (i.e. group_n = nb/num_xcds)
was NEVER swept. grouped_xcd_pid applies the XCD remap (pid%nx grouping) BEFORE the
band swizzle, so when #bands == nx each XCD's pid-block maps to exactly ONE N-band ->
that XCD's CUs share one B-stripe in their L2 slice (max reuse). production picks
#bands=8=nx (gn=nb//8). But the OPTIMAL band count may be 4 (wider bands, more A
reuse per band) or 16 (narrower bands, smaller B working-set) -- untested.

ANCHOR = production (nx8, gn=nb//8). Candidates = aligned (nx4, gn=nb//4),
(nx16, gn=nb//16) [both keep band==XCD alignment], + the MIS-aligned cross
(nx8, gn=nb//4) and (nx8, gn=nb//16) for contrast (these are r63-style, expected
worse). Pure tile->CU permutation => bit-exact (spot-checked maxdiff=0); only perf
in question. A reliable >=1% winner (>=7/T pairs AND cross-M consistent) routes
per-shape into recommend_config (ACCEPT); else ROLLBACK (8 bands IS optimal, dim
closed under the gold-standard joint protocol).
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


def build(M, N, K, a, b, asc, bsc, gn, nx, BM=256, BN=256):
    asp = ascale(asc, M, K, BM)
    bsp = preshuffle_scale_b_comb(bsc, K).view(-1)
    c = torch.zeros((M, N), dtype=torch.bfloat16, device="cuda")
    st = torch.cuda.current_stream()
    fn = compile_mxfp4_gemm_8w(K=K, BLOCK_M=BM, BLOCK_N=BN, mode="pipe",
                               group_m=4, group_n=gn, num_xcds=nx)
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


T = 8
SHAPES = [("70Bgateup", 28672, 8192), ("70Bdown", 8192, 28672), ("7Bdown", 4096, 11008)]
for tag, N, K in SHAPES:
    for M in (4096, 8192):
        nb = N // 256
        gn0 = nb // 8  # production anchor group_n (= 14 / 4 / 2)
        a, b, asc, bsc = mkraw(M, N, K)
        tf = lambda ms: 2 * M * N * K / (ms / 1e6) / 1e12
        ac, aar, abase = build(M, N, K, a, b, asc, bsc, gn0, 8)  # anchor (nx8, gn=nb//8)
        base = abase.clone()
        # aligned diagonal (#bands==nx) + mis-aligned contrast (nx8 off-gn)
        cands = []
        if nb // 4 >= 1:
            cands.append((nb // 4, 4))    # aligned: 4 wide bands, nx4
        if nb // 16 >= 1:
            cands.append((nb // 16, 16))  # aligned: 16 narrow bands, nx16
        cands.append((nb // 4, 8))        # mis-aligned contrast (r63-style)
        if nb // 16 >= 1:
            cands.append((nb // 16, 8))   # mis-aligned contrast
        seen = set([(gn0, 8)]); cands = [c for c in cands if not (c in seen or seen.add(c))]
        print(f"=== {tag} M={M} N={N} K={K} nb={nb}  anchor=(nx8,gn{gn0})={tf(bench(ac, aar)):.0f}TF ===", flush=True)
        # bit-exact spot check (pure permutation => 0)
        for (gn, nx) in cands[:2]:
            cx, arx, cc_ = build(M, N, K, a, b, asc, bsc, gn, nx)
            md = (cc_.float() - base.float()).abs().max().item()
            print(f"  bit-diff (nx{nx},gn{gn}) vs anchor maxdiff={md:.4g} (perm=>0)", flush=True)
        for (gn, nx) in cands:
            cc, car, _ = build(M, N, K, a, b, asc, bsc, gn, nx)
            wins = 0; ta = []; tc = []
            for t in range(T):
                t_a = bench(ac, aar); t_c = bench(cc, car)
                ta.append(t_a); tc.append(t_c)
                wins += (t_c < t_a)
            ma = st_.median(ta); mc = st_.median(tc)
            flag = "  <-- WIN?" if (mc < ma * 0.99 and wins >= 7) else ""
            print(f"  (nx{nx:2d},gn{gn:2d}): {tf(mc):6.0f}TF vs {tf(ma):6.0f}TF  ratio={tf(mc)/tf(ma):.3f}  wins {wins}/{T}{flag}", flush=True)
print("DONE", flush=True)
