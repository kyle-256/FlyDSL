"""round-54 (Mode-B B10 r10_1, perf leg): interleaved BM128 vs BM192/BM256 on kv.

Correctness+det0 already PASS (staged + pipe, SNR 55.6 + DET0). This is the
GO/refute perf gate. kv M4096: BM256=128wg, BM192=176wg (current route),
BM128=256wg (FULL occupancy). Does the +45% wg (176->256) beat BM128's 2x
M-tile prelude overhead? (BN64 r21 failed this exact test: more wg but 2x
overhead + halved N-parallelism -> net loss; BM128 keeps full BN128 N-parallelism
like BM192 did, so the failure mode may NOT transfer.) M8192: BM256=256wg-full,
BM128=512wg overshoot -> expected regress (gating would exclude, like BM192).

Thermal-matched interleaved, median-of-10, win-count (campaign hard lesson:
sub-ms kv shapes need interleaved A/B, not block-of-N).
"""
import sys, torch, statistics as st_
_R = "/workspace/code/FlyDSL"
for p in (_R, _R + "/flydsl/src"):
    sys.path.insert(0, p)
import flydsl.compiler as flyc
from turbo.mxfp4_gemm_8wave import compile_mxfp4_gemm_8w
from turbo.mxfp8_gemm_8wave import preshuffle_scale
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


def build(M, N, K, BM, BN, a, b, asc, bsc, gm):
    asp = ascale(asc, M, K, BM)
    bsp = preshuffle_scale(bsc, K, BN // 128).view(-1)
    c = torch.zeros((M, N), dtype=torch.bfloat16, device="cuda")
    st = torch.cuda.current_stream()
    fn = compile_mxfp4_gemm_8w(K=K, BLOCK_M=BM, BLOCK_N=BN, mode="pipe", group_m=gm, group_n=0)
    ar = (a, b, c.view(-1), asp, bsp, M, N, st)
    cc = flyc.compile(fn, *ar)
    cc(*ar); torch.cuda.synchronize()
    return cc, ar


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


N, K = 1024, 8192
T = 10


def tf(M, ms):
    return 2 * M * N * K / (ms / 1e6) / 1e12


# kv M4096: BM128(256wg) vs BM192(176wg, current route) vs BM256(128wg)
for M in (4096, 8192):
    a, b, asc, bsc = mkraw(M, N, K)
    cc128, ar128 = build(M, N, K, 128, 128, a, b, asc, bsc, 2)
    cc192, ar192 = build(M, N, K, 192, 128, a, b, asc, bsc, 2)
    cc256, ar256 = build(M, N, K, 256, 128, a, b, asc, bsc, 2)
    cur = cc192 if M == 4096 else cc256          # current production route per shape
    arcur = ar192 if M == 4096 else ar256
    curname = "BM192" if M == 4096 else "BM256"
    t128s, tcurs = [], []
    wins = 0
    for t in range(T):
        t128 = bench(cc128, ar128); tc = bench(cur, arcur)
        t128s.append(t128); tcurs.append(tc)
        if t128 < tc:
            wins += 1
    m128 = st_.median(t128s); mcur = st_.median(tcurs)
    print(f"kv M={M}: BM128={tf(M,m128):6.0f}TF  cur({curname})={tf(M,mcur):6.0f}TF  "
          f"BM128/cur={tf(M,m128)/tf(M,mcur):.3f}  BM128 wins {wins}/{T}", flush=True)
print("DONE", flush=True)
