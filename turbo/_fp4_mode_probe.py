"""round-72 probe: does a runtime (non-unrolled) K-loop variant (rt/rt2/il) —
which lets the LLVM scheduler interleave S2R/G2S among the MFMAs (the
'native LLIR-scheduler mechanism') — beat the hand-unrolled production `pipe`
on the bulk (MFMA-idle / control-bound) shapes? Correctness gate = SNR vs the
validated `pipe` output; perf = hot best-of-3 (same protocol as _fp4_vs_aiter)."""
import sys, os, torch
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


def run_mode(M, N, K, mode):
    d = "cuda"
    g = torch.Generator(device=d).manual_seed(1234)  # SAME data across modes
    a = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device=d, generator=g)
    b = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=d, generator=g)
    asc = torch.randint(125, 130, (M, K // SB), dtype=torch.uint8, device=d, generator=g)
    bsc = torch.randint(125, 130, (N, K // SB), dtype=torch.uint8, device=d, generator=g)
    c = torch.zeros((M, N), dtype=torch.bfloat16, device=d)
    asp = preshuffle_scale(asc, K, 4); bsp = preshuffle_scale_b_comb(bsc, K)
    ai = a.view(torch.int8).view(-1); bi = b.view(torch.int8).view(-1); st = torch.cuda.current_stream()
    fn = compile_mxfp4_gemm_8w(K=K, BLOCK_M=256, BLOCK_N=256, mode=mode, group_m=4, group_n=0)
    ar = (ai, bi, c.view(-1), asp.view(-1), bsp.view(-1), M, N, st)
    cc = flyc.compile(fn, *ar); cc(*ar); torch.cuda.synchronize()
    out = c.clone()
    us = bench(lambda: cc(*ar))
    tf = 2 * M * N * K / (us / 1e6) / 1e12
    return tf, out


# bulk BN256 shapes (control-bound regime where rt/il could help)
SHAPES = [
    ("70B q/o", 8192, 8192), ("70B down", 8192, 28672),
    ("70B gate/up", 28672, 8192), ("7B q/o", 4096, 4096),
]
MODES = ["pipe", "rt", "rt2", "il"]
M = 8192
print(f"M={M}  (pipe = production reference for SNR + perf ratio)")
print(f"{'shape':12s} {'N':>6s} {'K':>6s} | " + " ".join(f"{m+'_TF':>9s}" for m in MODES)
      + " | " + " ".join(f"{m+'_SNR':>8s}" for m in MODES[1:]) + " | best_vs_pipe")
for tag, N, K in SHAPES:
    tfs = {}; outs = {}
    for m in MODES:
        try:
            tf, out = run_mode(M, N, K, m)
            tfs[m] = tf; outs[m] = out
        except Exception as e:
            tfs[m] = None; outs[m] = None
            print(f"  [{tag} mode={m}] FAILED: {repr(e)[:160]}")
    ref = outs["pipe"]
    snrs = {m: (snr(outs[m], ref) if outs.get(m) is not None and ref is not None else -99) for m in MODES[1:]}
    pipe_tf = tfs["pipe"] or 1e-9
    cand = [(m, tfs[m]) for m in MODES[1:] if tfs.get(m) and snrs[m] >= 40]
    best = max(cand, key=lambda x: x[1]) if cand else ("none", 0)
    tfstr = " ".join(f"{(tfs[m] if tfs.get(m) else 0):9.0f}" for m in MODES)
    snrstr = " ".join(f"{snrs[m]:8.1f}" for m in MODES[1:])
    bestr = f"{best[0]} {best[1]/pipe_tf:.3f}" if best[1] else "none-correct"
    print(f"{tag:12s} {N:6d} {K:6d} | {tfstr} | {snrstr} | {bestr}")
