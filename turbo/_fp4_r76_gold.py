"""round-76 GOLD confirm (T=15 fresh-data interleaved) of the two cross-M-consistent
aligned-joint candidates surfaced by _fp4_r76_nxgn_joint.py:
  (A) 7B down  (N=4096,K=11008,nb=16): nx8 group_n 2->4  (T=8 was +1.6%/+1.0% 8/8 both M)
  (B) 70B down (N=8192,K=28672,nb=32): (nx8,gn4) anchor -> (nx16,gn2) joint  (+1.9%/+0.5% 8/8)
Both bit-exact (tile->CU permutation). GOLD standard = the r61 accept bar:
T=15 trials, FRESH random input bytes each trial, interleaved same-process, win-count
vs anchor. ACCEPT a candidate iff ratio>=1.01 AND >=11/15 AND cross-M direction positive."""
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
    return cc, ar


def refill(ar, M, N, K):
    # fresh random bytes in-place (a,b are ar[0],ar[1]); scales fixed (perm-invariant)
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
# (tag, N, K, anchor(gn,nx), cand(gn,nx))
CASES = [
    ("7Bdown ", 4096, 11008, (2, 8), (4, 8)),
    ("70Bdown", 8192, 28672, (4, 8), (2, 16)),
]
for tag, N, K, (agn, anx), (cgn, cnx) in CASES:
    for M in (4096, 8192):
        a, b, asc, bsc = mkraw(M, N, K)
        tf = lambda ms: 2 * M * N * K / (ms / 1e6) / 1e12
        ac, aar = build(M, N, K, a, b, asc, bsc, agn, anx)
        cc, car = build(M, N, K, a, b, asc, bsc, cgn, cnx)
        wins = 0; ta = []; tc = []
        for t in range(T):
            refill(aar, M, N, K); car[0].copy_(aar[0]); car[1].copy_(aar[1])  # same fresh data both
            t_a = bench(ac, aar); t_c = bench(cc, car)
            ta.append(t_a); tc.append(t_c); wins += (t_c < t_a)
        ma = st_.median(ta); mc = st_.median(tc); r = tf(mc) / tf(ma)
        ok = "ACCEPT" if (r >= 1.01 and wins >= 11) else ("marginal" if r >= 1.005 else "noise")
        print(f"{tag} M={M}: anchor(nx{anx},gn{agn})={tf(ma):.0f}TF -> cand(nx{cnx},gn{cgn})={tf(mc):.0f}TF  ratio={r:.3f}  wins {wins}/{T}  [{ok}]", flush=True)
print("DONE", flush=True)
