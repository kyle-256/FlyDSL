"""Round-42 (fix-pipeh r_h2): pipeh PERF GO/NO-GO gate (times pipeh DESPITE broken SNR).

Rationale: pipeh is the only barrier-light kernel variant (7 barrier + 4 setprio vs
pipe 29+23) attacking the bulk control:mfma ~50% ceiling (r34). r40/r41 found it
SNR-broken in the steady-state main loop and queued a multi-session correctness debug
(r_h2 bisection). BUT the ROI of that debug was never measured: even a CORRECT pipeh's
perf upside is unverified.

Key insight: a numerically-broken pipeh still issues the SAME instruction stream
(same K-iter count, same MFMA/barrier/ds_read/buffer_load mix) — its wrongness comes
from a missing hazard wait, not from skipped work. Fixing the hazard can only ADD
waits/barriers, so:

    time(broken-pipeh)  <=  time(fixed-pipeh)      [broken = speed UPPER BOUND]

Therefore if broken-pipeh is NOT faster than pipe on bulk, fixed-pipeh CANNOT be either
-> the entire multi-session fix-pipeh project has no ROI -> kill decisively (Rule-7
comparative gate before a multi-session commit, like r_5b SOTA-disasm that killed B5).

This script: route each shape via recommend_config (same cfg both modes; only mode
differs = single variable), build pipe + pipeh, report SNR for the record, then bench
BOTH regardless of SNR via thermal-matched interleaved trials -> median pipeh/pipe ratio
+ geomean over the bulk subset. GPU7 only.
"""
import os, sys, math, statistics, torch
_R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for p in (_R, os.path.join(_R, "flydsl", "src")):
    if p not in sys.path:
        sys.path.insert(0, p)
import flydsl.compiler as flyc
from flydsl.runtime.device import get_rocm_arch
from tests.kernels.utils import fp4_utils
from turbo.mxfp4_gemm_8wave import compile_mxfp4_gemm_8w, recommend_config
from turbo.mxfp8_gemm_8wave import preshuffle_scale, preshuffle_scale_b_comb
SB = 32


def ascale(asc, M, K, BM):
    nta = BM // 64; q = 16 * nta
    pad = ((M + q - 1) // q) * q
    if pad != M:
        ap = torch.zeros((pad, K // SB), dtype=torch.uint8, device=asc.device); ap[:M] = asc; asc = ap
    return preshuffle_scale(asc, K, nta).view(-1)


def bscale(bsc, K, BN):
    return (preshuffle_scale_b_comb(bsc, K) if BN >= 256 else preshuffle_scale(bsc, K, BN // 128)).view(-1)


def snr_db(out, ref):
    out, ref = out.float(), ref.float()
    n = (out - ref).pow(2).mean(); s = ref.pow(2).mean()
    return float("inf") if n.item() == 0 else (10 * torch.log10(s / n)).item()


def mk(M, N, K, dev="cuda"):
    a = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device=dev)
    b = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=dev)
    asc = torch.randint(124, 131, (M, K // SB), dtype=torch.uint8, device=dev)
    bsc = torch.randint(124, 131, (N, K // SB), dtype=torch.uint8, device=dev)
    return a, b, asc, bsc


def build(M, N, K, BM, BN, gm, gn, mode):
    fn = compile_mxfp4_gemm_8w(K=K, BLOCK_M=BM, BLOCK_N=BN, mode=mode, group_m=gm, group_n=gn)
    def args(a, b, c, sa, sb):
        return (a.view(torch.int8).view(-1), b.view(torch.int8).view(-1), c.view(-1),
                sa, sb, M, N, torch.cuda.current_stream())
    return fn, args


def time_once(fn, it=50):
    e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(); e0.record()
    for _ in range(it): fn()
    e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1) * 1000 / it  # us/iter


# bulk = compiler-scheduler-bound shapes pipeh targets (square + big-K); gate driver
BULK = {"7B q/o", "7B down", "70B q/o", "70B down"}
SHAPES = [("7B q/o", 4096, 4096), ("7B gate/up", 11008, 4096), ("7B down", 4096, 11008),
          ("70B q/o", 8192, 8192), ("70B kv", 1024, 8192),
          ("70B gate/up", 28672, 8192), ("70B down", 8192, 28672)]


def main():
    arch = str(get_rocm_arch()); assert "gfx95" in arch, f"needs gfx950, got {arch}"
    dev = "cuda"
    print(f"{'shape':12s}{'M':>6}{'N':>6}{'K':>6}  {'cfg':>16s}  {'SNRpipe':>8s}{'SNRpipeh':>9s}  "
          f"{'pipe_us':>8s}{'pipeh_us':>9s}  {'ratio':>6s}  bulk")
    lr = 0.0; n = 0; nwin = 0
    for M in (4096, 8192):
        for tag, N, K in SHAPES:
            BM, BN, gm, gn = recommend_config(M, N, K)
            a, b, asc, bsc = mk(M, N, K, dev)
            ref = torch.matmul(
                fp4_utils.mxfp4_to_f32(a)[:M, :K].float() * fp4_utils.e8m0_to_f32(asc).repeat_interleave(SB, -1)[:M, :K].float(),
                (fp4_utils.mxfp4_to_f32(b)[:N, :K].float() * fp4_utils.e8m0_to_f32(bsc).repeat_interleave(SB, -1)[:N, :K].float()).T)
            sa = ascale(asc, M, K, BM); sb = bscale(bsc, K, BN)
            cc = {}; snr = {}
            try:
                for mode in ("pipe", "pipeh"):
                    c = torch.zeros((M, N), dtype=torch.bfloat16, device=dev)
                    fn, mka = build(M, N, K, BM, BN, gm, gn, mode)
                    ar = mka(a, b, c, sa, sb)
                    cf = flyc.compile(fn, *ar); cf(*ar); torch.cuda.synchronize()
                    snr[mode] = snr_db(c, ref); cc[mode] = (cf, ar)
            except Exception as e:
                print(f"{tag:12s}{M:>6}{N:>6}{K:>6}  {str((BM,BN,gm,gn)):>16s}  ERR {type(e).__name__}: {e}")
                continue
            fp = lambda: cc["pipe"][0](*cc["pipe"][1])
            fh = lambda: cc["pipeh"][0](*cc["pipeh"][1])
            # warm both
            for _ in range(10): fp(); fh()
            torch.cuda.synchronize()
            # thermal-matched interleaved trials -> median ratio (>1 = pipeh faster)
            ratios = []; pus = []; hus = []
            for _ in range(7):
                tp = time_once(fp); th = time_once(fh)  # alternate, cancels drift
                ratios.append(tp / th); pus.append(tp); hus.append(th)
            r = statistics.median(ratios)
            isbulk = tag in BULK
            if isbulk:
                lr += math.log(r); n += 1; nwin += (r > 1.01)
            print(f"{tag:12s}{M:>6}{N:>6}{K:>6}  {str((BM,BN,gm,gn)):>16s}  {snr['pipe']:>8.1f}{snr['pipeh']:>9.1f}  "
                  f"{statistics.median(pus)*1e3:>8.1f}{statistics.median(hus)*1e3:>9.1f}  {r:>6.3f}  {'BULK' if isbulk else ''}")
    if n:
        g = math.exp(lr / n)
        print(f"\nGATE: pipeh/pipe BULK geomean = {g:.4f} over {n} bulk shapes; pipeh wins(>1.01): {nwin}/{n}")
        print(f"  (broken-pipeh time = speed UPPER BOUND for fixed-pipeh; fixing only ADDS waits)")
        if g > 1.03:
            print("  -> GO: broken-pipeh already faster -> a correct pipeh could win -> fix-pipeh worth continuing")
        else:
            print("  -> NO-GO: broken-pipeh NOT faster -> fixed-pipeh (slower) cannot beat pipe -> KILL fix-pipeh project")


if __name__ == "__main__":
    main()
