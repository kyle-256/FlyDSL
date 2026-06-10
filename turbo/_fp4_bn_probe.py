"""round-88 probe (Mode-B occupancy lens, distinct from r87 stage-count): does
BLOCK_N=128 beat production BLOCK_N=256 on the bulk shapes?  Motivation = r85 PMC:
bulk occupancy 5.46%, VGPR-limited at 228; the DOMINANT VGPR consumer is the
accumulators (128 f32/lane for BLOCK 256x256, N_ACCUMS=8).  BN128 halves N_ACCUMS
(8->4) => acc VGPR 128->64 — a far bigger VGPR drop than r87's stage-count
(which only freed ~16-32 next-frags, and lost to latency-hiding).  If occupancy is
truly the binding constraint, halving the dominant VGPR consumer is the strongest
occupancy lever.  BN128 is correctness-sealed (kv routes to it, det0 r83) but its
PERF on bulk was never measured (routing gate only selects BN128 for grid-underfill
kv).  Risk (r21 BN64 analogue): smaller N-tile => 2x grid + halved N-parallelism +
2x prelude ratio could make it memory/overhead-bound and lose.
Single variable = BLOCK_N (256 vs 128), BM=256/gm=4/gn=0/nx=8 held fixed.
Correctness gate = SNR(BN128 vs BN256), perf = hot best-of-3."""
import sys, torch
_R = "/workspace/code/FlyDSL"
for p in (_R, _R + "/flydsl/src"):
    sys.path.insert(0, p)
import flydsl.compiler as flyc
from turbo.mxfp4_gemm_8wave import compile_mxfp4_gemm_8w
from turbo.mxfp8_gemm_8wave import preshuffle_scale, preshuffle_scale_b_comb
SB = 32


def bench(fn, it=30, wu=10, reps=3):
    fn(); torch.cuda.synchronize()
    for _ in range(wu):
        fn()
    torch.cuda.synchronize(); best = 1e9
    for _ in range(reps):
        e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(); e0.record()
        for _ in range(it):
            fn()
        e1.record(); torch.cuda.synchronize(); best = min(best, e0.elapsed_time(e1) * 1000 / it)
    return best


def snr(o, r):
    o = o.float(); r = r.float()
    p = (r * r).mean().item(); n = ((o - r) ** 2).mean().item()
    return 99.0 if n == 0 else 10.0 * (torch.log10(torch.tensor(p / max(n, 1e-30)))).item()


def ascale(asc, M, K, BM):
    nta = BM // 64
    q = 16 * nta
    pad = ((M + q - 1) // q) * q
    if pad != M:
        ap = torch.zeros((pad, K // SB), dtype=torch.uint8, device=asc.device)
        ap[:M] = asc
        asc = ap
    return preshuffle_scale(asc, K, nta).view(-1)


def run_bn(M, N, K, BN):
    d = "cuda"
    g = torch.Generator(device=d).manual_seed(1234)  # SAME data across BN
    a = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device=d, generator=g)
    b = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=d, generator=g)
    asc = torch.randint(125, 130, (M, K // SB), dtype=torch.uint8, device=d, generator=g)
    bsc = torch.randint(125, 130, (N, K // SB), dtype=torch.uint8, device=d, generator=g)
    c = torch.zeros((M, N), dtype=torch.bfloat16, device=d)
    BM = 256
    asp = ascale(asc, M, K, BM)
    bsp = (preshuffle_scale_b_comb(bsc, K) if BN >= 256 else preshuffle_scale(bsc, K, BN // 128)).view(-1)
    ai = a.view(torch.int8).view(-1); bi = b.view(torch.int8).view(-1); st = torch.cuda.current_stream()
    fn = compile_mxfp4_gemm_8w(K=K, BLOCK_M=BM, BLOCK_N=BN, mode="pipe", group_m=4, group_n=0, num_xcds=8)
    ar = (ai, bi, c.view(-1), asp, bsp, M, N, st)
    cc = flyc.compile(fn, *ar); cc(*ar); torch.cuda.synchronize()
    out = c.clone()
    us = bench(lambda: cc(*ar))
    tf = 2 * M * N * K / (us / 1e6) / 1e12
    return tf, out


SHAPES = [
    ("70B q/o", 8192, 8192), ("70B down", 8192, 28672),
    ("70B gate/up", 28672, 8192), ("7B q/o", 4096, 4096),
]
for M in (8192, 4096):
    print(f"\n==== M={M}  (BN256 = production reference) ====")
    print(f"{'shape':12s} {'N':>6s} {'K':>6s} | {'BN256_TF':>10s} {'BN128_TF':>10s} | BN128_SNR | BN128/BN256")
    for tag, N, K in SHAPES:
        try:
            tf256, o256 = run_bn(M, N, K, 256)
            tf128, o128 = run_bn(M, N, K, 128)
            s = snr(o128, o256)
            ratio = tf128 / tf256 if tf256 else 0
            print(f"{tag:12s} {N:6d} {K:6d} | {tf256:10.0f} {tf128:10.0f} | {s:9.1f} | {ratio:.3f}")
        except Exception as e:
            print(f"  [{tag}] FAILED: {repr(e)[:160]}")
