"""Round-31 (B6 r6_3): port narrow-A-G2S into the production PIPE kernel for BM192.
Validates:
  (1) pipe BM192/BN128 correctness vs staged BM256/BN256 ref on kv (SNR + det2)
  (2) det0 300-run fresh kv M4096 pipe BM192/BN128
  (3) BM256 byte-identical guard: pipe BM256/BN128 (production kv route) unchanged
  (4) PERF GO/kill: interleaved thermal-matched pipe-BM256/BN128 vs pipe-BM192/BN128
      on kv M4096 (occupancy-bound) + M8192 (already 256wg-full, must not regress).
kv group_m=2 (production routing); BN128 both configs (apples-to-apples occupancy).
"""
import sys, torch
_R = "/workspace/code/FlyDSL"
for p in (_R, _R + "/flydsl/src"):
    sys.path.insert(0, p)
import flydsl.compiler as flyc
from turbo.mxfp4_gemm_8wave import compile_mxfp4_gemm_8w
from turbo.mxfp8_gemm_8wave import preshuffle_scale, preshuffle_scale_b_comb
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


def build(M, N, K, BM, BN, mode, a, b, asc, bsc, gm):
    asp = ascale(asc, M, K, BM)
    bsp = (preshuffle_scale_b_comb(bsc, K) if BN >= 256 else preshuffle_scale(bsc, K, BN // 128)).view(-1)
    c = torch.zeros((M, N), dtype=torch.bfloat16, device="cuda")
    st = torch.cuda.current_stream()
    fn = compile_mxfp4_gemm_8w(K=K, BLOCK_M=BM, BLOCK_N=BN, mode=mode, group_m=gm, group_n=0)
    ar = (a, b, c.view(-1), asp, bsp, M, N, st)
    cc = flyc.compile(fn, *ar)
    cc(*ar); torch.cuda.synchronize()
    return c.clone(), cc, ar


def bench(cc, ar, it=30, wu=12, reps=3):
    cc(*ar); torch.cuda.synchronize()
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

# (1) correctness: pipe BM192/BN128 vs staged BM256/BN256 ref
print("=== (1) pipe BM192/BN128 correctness vs staged BM256/BN256 ref (kv) ===", flush=True)
for M in (768, 4096):
    a, b, asc, bsc = mkraw(M, N, K)
    ref, _, _ = build(M, N, K, 256, 256, "staged", a, b, asc, bsc, 4)
    out, cc, ar = build(M, N, K, 192, 128, "pipe", a, b, asc, bsc, 2)
    s = snr(out, ref)
    c2 = torch.zeros((M, N), dtype=torch.bfloat16, device="cuda")
    ar2 = (ar[0], ar[1], c2.view(-1), ar[3], ar[4], M, N, ar[7])
    cc(*ar2); torch.cuda.synchronize()
    det2 = torch.equal(out, c2)
    print(f"  M={M:5d} pipe BM192/BN128 vs ref: SNR={s:6.1f} dB  det2={'OK' if det2 else 'NO'}", flush=True)

# (3) BM256 byte-identical guard: pipe BM256/BN128 (production kv route) still correct
print("=== (3) BM256/BN128 pipe production-route guard (vs staged BM256/BN256) ===", flush=True)
M = 4096
a, b, asc, bsc = mkraw(M, N, K)
ref, _, _ = build(M, N, K, 256, 256, "staged", a, b, asc, bsc, 4)
out, _, _ = build(M, N, K, 256, 128, "pipe", a, b, asc, bsc, 2)
print(f"  M={M} pipe BM256/BN128 vs ref: SNR={snr(out, ref):6.1f} dB", flush=True)

# (2) det0 300-run fresh kv M4096 pipe BM192/BN128
print("=== (2) det0 300-run fresh kv M4096 pipe BM192/BN128 ===", flush=True)
M = 4096
fn = compile_mxfp4_gemm_8w(K=K, BLOCK_M=192, BLOCK_N=128, mode="pipe", group_m=2, group_n=0)
maxd = 0.0; nanflag = False
for run in range(300):
    a, b, asc, bsc = mkraw(M, N, K)
    asp = ascale(asc, M, K, 192)
    bsp = preshuffle_scale(bsc, K, 1).view(-1)
    c1 = torch.zeros((M, N), dtype=torch.bfloat16, device="cuda")
    st = torch.cuda.current_stream()
    ar = (a, b, c1.view(-1), asp, bsp, M, N, st)
    cc = flyc.compile(fn, *ar)
    cc(*ar); torch.cuda.synchronize(); o1 = c1.clone()
    c1.zero_(); cc(*ar); torch.cuda.synchronize()
    maxd = max(maxd, (o1.float() - c1.float()).abs().max().item())
    if torch.isnan(c1).any():
        nanflag = True
print(f"  det0: 300-run maxdiff={maxd} nan={nanflag} -> {'DET0' if maxd == 0 and not nanflag else 'FAIL'}", flush=True)

# (4) PERF GO/kill: interleaved thermal-matched pipe BM256 vs BM192 (BN128, gm2)
print("=== (4) PERF interleaved BM256/BN128 vs BM192/BN128 (kv, gm2) ===", flush=True)
for M in (4096, 8192):
    a, b, asc, bsc = mkraw(M, N, K)
    _, cc256, ar256 = build(M, N, K, 256, 128, "pipe", a, b, asc, bsc, 2)
    _, cc192, ar192 = build(M, N, K, 192, 128, "pipe", a, b, asc, bsc, 2)
    wins = 0; T = 10; t256s = []; t192s = []
    for t in range(T):
        t256 = bench(cc256, ar256, reps=2); t192 = bench(cc192, ar192, reps=2)
        t256s.append(t256); t192s.append(t192)
        if t192 < t256:
            wins += 1
    import statistics as st_
    m256 = st_.median(t256s); m192 = st_.median(t192s)
    tf256 = 2 * M * N * K / (m256 / 1e6) / 1e12
    tf192 = 2 * M * N * K / (m192 / 1e6) / 1e12
    print(f"  kv M={M}: BM256={tf256:6.0f}TF BM192={tf192:6.0f}TF  BM192/BM256={tf192/tf256:.3f}  "
          f"BM192 wins {wins}/{T} pairs", flush=True)
print("DONE", flush=True)
