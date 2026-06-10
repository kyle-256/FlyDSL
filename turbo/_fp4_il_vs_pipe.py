"""Round-26 probe: mode="il" sched=True (1-barrier double-buffer + explicit
structured sched_mfma/dsrd/vmem cadence = FlyDSL-native analog of aiter hand-asm
interleave, "no AGPR copy storm unlike pipe") vs production mode="pipe"
(s_setprio + 6-barrier/iter, manual mma).

Hypothesis: the il scheduler relieves the bulk MFMA-idle (48%) that the s_setprio
6-barrier pipe suffers -> narrows the bulk gap round-25 declared compiler-bound.
This is the closest in-kernel-scope analog to aiter's hand-asm instruction
scheduling and has NOT been benchmarked vs production pipe (R44/R108/C8 tested
ad-hoc sched_group_barrier masks ON the pipe; R4/pipe2 tested a different lean
減-barrier variant; never this full il double-buffer + structured-cadence kernel).

Correctness: pipe output is golden; il must be SNR>=40 dB. Bench: paired rotating
hot median (20 warmup + 7 rounds). Same recommend_config (BM,BN,gn) for both so
il inherits band swizzle + BN128 routing.
"""
import sys, os, math, statistics
import torch
_R = "/workspace/code/FlyDSL"
for p in (_R, _R + "/flydsl/src"):
    sys.path.insert(0, p)
import flydsl.compiler as flyc
from turbo.mxfp4_gemm_8wave import compile_mxfp4_gemm_8w, recommend_config
from turbo.mxfp8_gemm_8wave import preshuffle_scale, preshuffle_scale_b_comb

SB = 32


def make_inputs(M, N, K):
    d = "cuda"
    a = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device=d)
    b = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=d)
    asc = torch.randint(125, 130, (M, K // SB), dtype=torch.uint8, device=d)
    bsc = torch.randint(125, 130, (N, K // SB), dtype=torch.uint8, device=d)
    c = torch.zeros((M, N), dtype=torch.bfloat16, device=d)
    asp = preshuffle_scale(asc, K, 4)
    bsp = preshuffle_scale_b_comb(bsc, K)
    ai = a.view(torch.int8).view(-1)
    bi = b.view(torch.int8).view(-1)
    st = torch.cuda.current_stream()
    ar = (ai, bi, c.view(-1), asp.view(-1), bsp.view(-1), M, N, st)
    return ar, c


def snr(o, r):
    diff = (o.float() - r.float())
    s = (r.float() ** 2).sum().item()
    n = (diff ** 2).sum().item()
    return 99.0 if n == 0 else 10 * math.log10(s / max(n, 1e-30))


def meas(cc, ar, it=40):
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    e0.record()
    for _ in range(it):
        cc(*ar)
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) * 1000 / it  # us


# NOTE: kernel_gemm_il uses range_constexpr (full unroll) over K_ITERS, so big-K
# shapes (70Bdown K=28672 -> 224 iters) are a compile-time bomb. Restrict to small
# K_ITERS bulk shapes; if il wins here, the big-K compile-cost itself is a finding.
SHAPES = [
    ("7Bqo",    4096, 4096, 4096),    # K_ITERS=32 square bulk <- target
    ("70Bqo",   8192, 8192, 8192),    # K_ITERS=64 square bulk <- target
]
# candidate: mode="il" sched=True ; golden: production mode="pipe"
VARIANTS = [
    ("pipe", dict(mode="pipe")),
    ("il_sched", dict(mode="il", sched=True)),
]
ROUNDS = 7

for tag, M, N, K in SHAPES:
    BM, BN, gn = recommend_config(M, N, K)
    cfg = (BM, BN, gn)
    ccs = {}
    for vn, kw in VARIANTS:
        try:
            fn = compile_mxfp4_gemm_8w(K=K, BLOCK_M=BM, BLOCK_N=BN, group_m=4, group_n=gn, **kw)
            ar, c = make_inputs(M, N, K)
            cc = flyc.compile(fn, *ar)
            cc(*ar)
            torch.cuda.synchronize()
            ccs[vn] = (cc, ar, c)
        except Exception as e:
            print(f"=== {tag} variant {vn} COMPILE/RUN FAILED: {type(e).__name__}: {str(e)[:200]}")
            ccs[vn] = (None, None, None)

    gold = ccs["pipe"][2].clone() if ccs["pipe"][0] is not None else None
    snrs = {}
    for vn, _ in VARIANTS:
        cc, ar, c = ccs[vn]
        if cc is None or gold is None:
            snrs[vn] = float("nan"); continue
        c.zero_(); cc(*ar); torch.cuda.synchronize()
        snrs[vn] = snr(c, gold)

    for vn, _ in VARIANTS:
        cc, ar, _ = ccs[vn]
        if cc is None: continue
        for _ in range(20): cc(*ar)
    torch.cuda.synchronize()

    samples = {vn: [] for vn, _ in VARIANTS}
    for r in range(ROUNDS):
        for vn, _ in VARIANTS:
            cc, ar, _ = ccs[vn]
            if cc is None: continue
            samples[vn].append(meas(cc, ar))

    print(f"=== {tag} M{M} N{N} K{K} cfg{cfg} ===")
    pvals = samples["pipe"]
    base = statistics.median(pvals) if pvals else float("nan")
    for vn, _ in VARIANTS:
        vals = samples[vn]
        if not vals:
            print(f"  {vn:9s} (failed)"); continue
        med = statistics.median(vals)
        tf = 2 * M * N * K / (med / 1e6) / 1e12
        ratio = base / med if med else float("nan")
        print(f"  {vn:9s} med={med:8.2f}us tf={tf:6.0f} ratio_vs_pipe={ratio:.4f} SNR={snrs[vn]:.1f}")
    sys.stdout.flush()
