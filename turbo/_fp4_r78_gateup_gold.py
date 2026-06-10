"""round-78 GOLD confirm (T=15 fresh-data interleaved) of the LAST un-confirmed
aligned-nx×gn candidate: 70B gate/up (N=28672, K=8192, nb=112) big-N band.

production = (group_n=nb//8=14, num_xcds=8) = 8 bands. The aligned diagonal
candidates (#bands==num_xcds): (gn7, nx16)=16 narrow bands, (gn28, nx4)=4 wide bands.
r76 T=8 sweep showed (gn7,nx16) only +0.2-0.3% (M4096 7/8, M8192 6/8, sub-1%) and
(gn28,nx4) LOSES. r61 lesson: a sub-1% / low-T conclusion must be GOLD-confirmed
(T=15 fresh-data interleaved). ACCEPT iff ratio>=1.01 AND >=11/15 AND cross-M
consistent; else ROLLBACK (production gn14/nx8 IS optimal, big-N aligned vein closed).
Bit-exact (tile->CU permutation) — correctness by construction (spot-checked)."""
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
    nta = BM // 64; q = 16 * nta; pad = ((M + q - 1) // q) * q
    if pad != M:
        ap = torch.zeros((pad, K // SB), dtype=torch.uint8, device=asc.device); ap[:M] = asc; asc = ap
    return preshuffle_scale(asc, K, nta).view(-1)


def build(M, N, K, a, b, asc, bsc, gn, nx):
    asp = ascale(asc, M, K, 256); bsp = preshuffle_scale_b_comb(bsc, K).view(-1)
    c = torch.zeros((M, N), dtype=torch.bfloat16, device="cuda"); st = torch.cuda.current_stream()
    fn = compile_mxfp4_gemm_8w(K=K, BLOCK_M=256, BLOCK_N=256, mode="pipe", group_m=4, group_n=gn, num_xcds=nx)
    ar = (a, b, c.view(-1), asp, bsp, M, N, st)
    cc = flyc.compile(fn, *ar); cc(*ar); torch.cuda.synchronize()
    return cc, ar, c


def refill(ar):
    ar[0].copy_(torch.randint(0, 256, ar[0].shape, dtype=torch.uint8, device="cuda").view(torch.int8))
    ar[1].copy_(torch.randint(0, 256, ar[1].shape, dtype=torch.uint8, device="cuda").view(torch.int8))


def bench(cc, ar, it=30, wu=6, reps=2):
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


T = 15
N, K = 28672, 8192
ANCHOR = (14, 8)   # production gn=nb//8, nx=8
CANDS = [(7, 16), (28, 4)]   # aligned diagonal: 16 narrow bands, 4 wide bands
for M in (4096, 8192):
    a, b, asc, bsc = mkraw(M, N, K)
    tf = lambda ms: 2 * M * N * K / (ms / 1e6) / 1e12
    ac, aar, abase = build(M, N, K, a, b, asc, bsc, *ANCHOR); base = abase.clone()
    for (gn, nx) in CANDS:
        cc, car, cc_out = build(M, N, K, a, b, asc, bsc, gn, nx)
        md = (cc_out.float() - base.float()).abs().max().item()
        wins = 0; ta = []; tc = []
        for t in range(T):
            refill(aar); car[0].copy_(aar[0]); car[1].copy_(aar[1])
            t_a = bench(ac, aar); t_c = bench(cc, car)
            ta.append(t_a); tc.append(t_c); wins += (t_c < t_a)
        ma = st_.median(ta); mc = st_.median(tc); r = tf(mc) / tf(ma)
        ok = "ACCEPT" if (r >= 1.01 and wins >= 11) else ("marginal" if r >= 1.005 else "noise")
        print(f"70Bgu M={M}: anchor(nx8,gn14)={tf(ma):.0f}TF -> cand(nx{nx},gn{gn})={tf(mc):.0f}TF  ratio={r:.3f}  wins {wins}/{T}  bitdiff={md:.3g}  [{ok}]", flush=True)
print("DONE", flush=True)
