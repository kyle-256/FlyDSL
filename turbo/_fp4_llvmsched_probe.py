"""Round-15 probe: LLVM amdgpu-sched-strategy=iterative-ilp on mxfp4 pipe kernel.

Paired rotating bench of 4 representative shapes x 3 compile variants:
  - bundled       : current production path (no compile_hints, bundled MLIR backend)
  - ext-maxocc    : external LLVM, amdgpu-sched-strategy=max-occupancy (LLVM default)
  - ext-ilp       : external LLVM, amdgpu-sched-strategy=iterative-ilp (candidate)

Decision logic for round-15: a commit (production -> external+iterative-ilp) is only
worthwhile if ext-ilp BEATS bundled (since flipping production makes it go external).
ext-maxocc isolates the external-backend confound (external default vs bundled default).
Correctness: bundled output is golden; ext variants must be SNR>=40 dB (scheduler-only,
should be ~bit-exact).
"""
import sys, os, math, statistics, contextlib
import torch
_R = "/workspace/code/FlyDSL"
for p in (_R, _R + "/flydsl/src"):
    sys.path.insert(0, p)
import flydsl.compiler as flyc
from flydsl.compiler.kernel_function import CompilationContext
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


SHAPES = [
    ("70Bdown", 8192, 8192, 28672),   # big-K bulk worst (MFMA-idle)
    ("70Bgu",   8192, 28672, 8192),   # big-N (band)
    ("70Bqo",   8192, 8192, 8192),    # square bulk
    ("70Bkv",   4096, 1024, 8192),    # occupancy (BN=128 route)
]
VARIANTS = [
    ("bundled", None),
    ("ext-maxocc", {"llvm_options": {"amdgpu-sched-strategy": "max-occupancy"}}),
    ("ext-ilp", {"llvm_options": {"amdgpu-sched-strategy": "iterative-ilp"}}),
]
ROUNDS = 8

for tag, M, N, K in SHAPES:
    ccs = {}
    cfg = None
    for vn, h in VARIANTS:
        BM, BN, gn = recommend_config(M, N, K)
        cfg = (BM, BN, gn)
        fn = compile_mxfp4_gemm_8w(K=K, BLOCK_M=BM, BLOCK_N=BN, mode="pipe", group_m=4, group_n=gn)
        ar, c = make_inputs(M, N, K)
        ctx = CompilationContext.compile_hints(h) if h else contextlib.nullcontext()
        try:
            with ctx:
                cc = flyc.compile(fn, *ar)
                cc(*ar)
                torch.cuda.synchronize()
        except Exception as e:
            print(f"=== {tag} variant {vn} COMPILE FAILED: {type(e).__name__}: {str(e)[:200]}")
            cc = None
        ccs[vn] = (cc, ar, c)

    gold = ccs["bundled"][0] and ccs["bundled"][2].clone()
    snrs = {}
    for vn, _ in VARIANTS:
        cc, ar, c = ccs[vn]
        if cc is None:
            snrs[vn] = float("nan")
            continue
        c.zero_()
        cc(*ar)
        torch.cuda.synchronize()
        snrs[vn] = snr(c, gold)

    for vn, _ in VARIANTS:
        cc, ar, _ = ccs[vn]
        if cc is None:
            continue
        for _ in range(20):
            cc(*ar)
    torch.cuda.synchronize()

    samples = {vn: [] for vn, _ in VARIANTS}
    for r in range(ROUNDS):
        for vn, _ in VARIANTS:
            cc, ar, _ = ccs[vn]
            if cc is None:
                continue
            samples[vn].append(meas(cc, ar))

    print(f"=== {tag} M{M} N{N} K{K} cfg{cfg} ===")
    bvals = samples["bundled"]
    base = statistics.median(bvals) if bvals else float("nan")
    for vn, _ in VARIANTS:
        vals = samples[vn]
        if not vals:
            print(f"  {vn:12s} (failed)")
            continue
        med = statistics.median(vals)
        tf = 2 * M * N * K / (med / 1e6) / 1e12
        print(f"  {vn:12s} med={med:8.2f}us tf={tf:6.0f} ratio_vs_bundled={base/med:.4f} SNR={snrs[vn]:.1f}")
    sys.stdout.flush()
