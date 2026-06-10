"""round-87 probe (Mode-B occupancy lens): does the 1-stage `staged` kernel
(SharedStorageFp4 = 4 LDS arrays, NO a_next/b_next double-buffer frags) beat the
production 2-stage `pipe` (SharedStorageFp4Pipe = 8 arrays) on the bulk BN256
shapes?  Motivation = r85 dynamic PMC: bulk is Issue+Dep-Wait bound *amplified by
5.46% wavefront occupancy* (VGPR-limited at 228).  staged drops the "next" frags
=> lower VGPR => potentially higher occupancy => more waves to hide the 513
load-bearing WG barriers.  pipe3 (3-stage) being slower (r8_3 0.868x) is
consistent with "more stages = more VGPR = lower occupancy = slower" => the
occupancy hypothesis predicts FEWER stages could win when async DMA already
overlaps the loads (r65 vmcnt(0)=1).  staged perf was NEVER benchmarked (r70
dismissed it by assertion, before r85's occupancy finding).
Correctness gate = SNR vs validated `pipe`; perf = hot best-of-3 (same protocol
as _fp4_vs_aiter).  Single variable = stage count (mode pipe vs staged)."""
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


SHAPES = [
    ("70B q/o", 8192, 8192), ("70B down", 8192, 28672),
    ("70B gate/up", 28672, 8192), ("7B q/o", 4096, 4096),
]
MODES = ["pipe", "staged"]
for M in (8192, 4096):
    print(f"\n==== M={M}  (pipe = production reference for SNR + perf ratio) ====")
    print(f"{'shape':12s} {'N':>6s} {'K':>6s} | " + " ".join(f"{m+'_TF':>10s}" for m in MODES)
          + " | staged_SNR | staged/pipe")
    for tag, N, K in SHAPES:
        tfs = {}; outs = {}
        for m in MODES:
            try:
                tf, out = run_mode(M, N, K, m)
                tfs[m] = tf; outs[m] = out
            except Exception as e:
                tfs[m] = None; outs[m] = None
                print(f"  [{tag} mode={m}] FAILED: {repr(e)[:160]}")
        ref = outs.get("pipe")
        s = snr(outs["staged"], ref) if outs.get("staged") is not None and ref is not None else -99
        pipe_tf = tfs.get("pipe") or 1e-9
        st_tf = tfs.get("staged") or 0
        ratio = st_tf / pipe_tf if st_tf else 0
        tfstr = " ".join(f"{(tfs[m] if tfs.get(m) else 0):10.0f}" for m in MODES)
        print(f"{tag:12s} {N:6d} {K:6d} | {tfstr} | {s:10.1f} | {ratio:.3f}")
