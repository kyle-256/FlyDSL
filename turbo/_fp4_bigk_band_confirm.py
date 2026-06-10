"""round-61 confirm: the joint sweep surfaced 70B down M4096 (gm4, gn=nb//8=4)
= +4.2% (10/10), which CONTRADICTS r28 ("big-K band hurts"). Per the r46 lesson
(contradictory sweep data must be resolved with higher-trial interleaved A/B
before routing), re-verify at high T with FRESH data each pair, both M, on BOTH
big-K bulk shapes (70B down N=8192 nb=32, 7B down N=4096 nb=16). The skill itself
says the 2D band is general ("big-K also +1%"), so r28's "hurt" is the suspect.

Route rule candidate: enable gn=nb//8 for big-K (K>=11008) wide-ish N (nb>=16),
not just nb>=96. ACCEPT only shapes that hold >=+1.5% AND >=12/15 wins AND do not
regress the other M of the same (N,K).
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


def ascale(asc, M, K, BM=256):
    nta = BM // 64
    q = 16 * nta
    pad = ((M + q - 1) // q) * q
    if pad != M:
        ap = torch.zeros((pad, K // SB), dtype=torch.uint8, device=asc.device)
        ap[:M] = asc
        asc = ap
    return preshuffle_scale(asc, K, nta).view(-1)


def build(M, N, K, a, b, asc, bsc, gn):
    asp = ascale(asc, M, K)
    bsp = preshuffle_scale_b_comb(bsc, K).view(-1)
    c = torch.zeros((M, N), dtype=torch.bfloat16, device="cuda")
    st = torch.cuda.current_stream()
    fn = compile_mxfp4_gemm_8w(K=K, BLOCK_M=256, BLOCK_N=256, mode="pipe", group_m=4, group_n=gn)
    ar = (a, b, c.view(-1), asp, bsp, M, N, st)
    cc = flyc.compile(fn, *ar); cc(*ar); torch.cuda.synchronize()
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


T = 15
SHAPES = [("70Bdown", 8192, 28672), ("7Bdown", 4096, 11008)]
for tag, N, K in SHAPES:
    nb = N // 256
    gn = nb // 8
    for M in (4096, 8192):
        tf = lambda ms: 2 * M * N * K / (ms / 1e6) / 1e12
        wins = 0; ta = []; tc = []
        for t in range(T):
            a, b, asc, bsc = mkraw(M, N, K)  # FRESH data each pair
            a_cc, a_ar, _ = build(M, N, K, a, b, asc, bsc, 0)
            c_cc, c_ar, _ = build(M, N, K, a, b, asc, bsc, gn)
            t_a = bench(a_cc, a_ar); t_c = bench(c_cc, c_ar)
            ta.append(t_a); tc.append(t_c); wins += (t_c < t_a)
        ma = st_.median(ta); mc = st_.median(tc)
        flag = "  <== ROUTE" if (mc < ma * 0.985 and wins >= 12) else ""
        print(f"{tag} M={M} N={N} K={K} nb={nb} gn={gn}: band {tf(mc):6.0f}TF vs gn0 {tf(ma):6.0f}TF  "
              f"ratio={tf(mc)/tf(ma):.3f}  band wins {wins}/{T}{flag}", flush=True)
print("DONE", flush=True)
