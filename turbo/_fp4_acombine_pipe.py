"""round-58 (Mode-B B11 r_a2): combined-128 A in production PIPE — correctness + perf.

r_a1 validated staged combined-A. r_a2 ports to kernel_gemm_pipe: each A tile is
prefetched as ONE combined-128 G2S (a_load0, fills LDS-adjacent a_*_0+a_*_1 all-8-
waves) mirroring the BN128 B b_g2s_full placement; the second-half a_load1 calls
become no-ops. Gated by compile param a_combine (True=combined, False=narrow r55).

Part 1: correctness — pipe BM128/BN128/gm8 combined-A vs dequant ref (SNR≥40) +
det0 (2 fresh passes × 200 run, the harness that caught B's merged-spill race r_k6).
Part 2: perf — interleaved A/B combined(a_combine=True) vs narrow(False), T=10,
win-count. The combined-A is a merged-spill buffer_load_lds (a_*_0 + SPILL a_*_1)
like B; the loop wait_barrier(1) may need re-tuning if it races (then r_a3).
"""
import sys, torch, statistics as st_
_R = "/workspace/code/FlyDSL"
for p in (_R, _R + "/flydsl/src"):
    sys.path.insert(0, p)
import flydsl.compiler as flyc
from tests.kernels.utils import fp4_utils
from turbo.mxfp4_gemm_8wave import compile_mxfp4_gemm_8w
from turbo.mxfp8_gemm_8wave import preshuffle_scale
SB = 32


def snr(o, r):
    o, r = o.float(), r.float()
    n = (o - r).pow(2).mean(); s = r.pow(2).mean()
    return float("inf") if n.item() == 0 else (10 * torch.log10(s / n)).item()


def ref_mxfp4(a_u8, b_u8, asc, bsc, M, N, K):
    a_f = fp4_utils.mxfp4_to_f32(a_u8)[:M, :K].float()
    b_f = fp4_utils.mxfp4_to_f32(b_u8)[:N, :K].float()
    a_s = fp4_utils.e8m0_to_f32(asc).repeat_interleave(SB, dim=-1)[:M, :K].float()
    b_s = fp4_utils.e8m0_to_f32(bsc).repeat_interleave(SB, dim=-1)[:N, :K].float()
    return torch.matmul(a_f * a_s, (b_f * b_s).T)


def mkraw(M, N, K):
    d = "cuda"
    a = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device=d)
    b = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=d)
    asc = torch.randint(125, 130, (M, K // SB), dtype=torch.uint8, device=d)
    bsc = torch.randint(125, 130, (N, K // SB), dtype=torch.uint8, device=d)
    return a, b, asc, bsc


def build(M, N, K, a, b, asc, bsc, ac):
    asp = preshuffle_scale(asc, K, 128 // 64)
    bsp = preshuffle_scale(bsc, K, 1)
    c = torch.zeros((M, N), dtype=torch.bfloat16, device="cuda")
    st = torch.cuda.current_stream()
    fn = compile_mxfp4_gemm_8w(K=K, BLOCK_M=128, BLOCK_N=128, mode="pipe", group_m=8, group_n=0, a_combine=ac)
    ar = (a.view(torch.int8).view(-1), b.view(torch.int8).view(-1), c.view(-1), asp.view(-1), bsp.view(-1), M, N, st)
    cc = flyc.compile(fn, *ar); cc(*ar); torch.cuda.synchronize()
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


def tf(M, N, K, ms):
    return 2 * M * N * K / (ms / 1e6) / 1e12


print("=== Part 1: combined-A pipe correctness + det0 (kv) ===", flush=True)
for M, N, K in [(4096, 1024, 8192), (8192, 1024, 8192)]:
    worst_md = 0.0; snr0 = None
    for p in range(2):
        a, b, asc, bsc = mkraw(M, N, K)
        ref = ref_mxfp4(a, b, asc, bsc, M, N, K)
        cc, ar, c = build(M, N, K, a, b, asc, bsc, True)
        if p == 0:
            snr0 = snr(c.float(), ref)
        base = c.clone()
        for _ in range(200):
            c.zero_(); cc(*ar); torch.cuda.synchronize()
            md = (c.float() - base.float()).abs().max().item()
            worst_md = max(worst_md, md)
            if md != 0.0:
                break
    det = "DET0" if worst_md == 0.0 else f"NONDET({worst_md:.4g})"
    ok = "PASS" if (snr0 >= 40 and worst_md == 0.0) else "FAIL"
    print(f"  kv M={M}: combined-A SNR={snr0:6.2f} {det} (2pass x 200run) -> {ok}", flush=True)

print("=== Part 2: perf interleaved combined(ac=1) vs narrow(ac=0), kv ===", flush=True)
T = 10
for M, N, K in [(4096, 1024, 8192), (8192, 1024, 8192)]:
    a, b, asc, bsc = mkraw(M, N, K)
    cc1, ar1, _ = build(M, N, K, a, b, asc, bsc, True)
    cc0, ar0, _ = build(M, N, K, a, b, asc, bsc, False)
    t1s, t0s = [], []; wins = 0
    for t in range(T):
        t1 = bench(cc1, ar1); t0 = bench(cc0, ar0)
        t1s.append(t1); t0s.append(t0)
        if t1 < t0:
            wins += 1
    m1 = st_.median(t1s); m0 = st_.median(t0s)
    print(f"  kv M={M}: combined={tf(M,N,K,m1):6.0f}TF narrow={tf(M,N,K,m0):6.0f}TF "
          f"combined/narrow={tf(M,N,K,m1)/tf(M,N,K,m0):.3f} combined wins {wins}/{T}", flush=True)
print("DONE", flush=True)
