"""Round-40: mode='pipe' vs mode='pipeh' (hoisted dense-MFMA-burst variant) A/B.

pipeh issues all 4 ds_reads at K-iter top then one high-prio MFMA burst with G2S
loads streaming underneath (mirrors aiter's all-operands-resident dense burst) ->
aimed at the bulk control-bound / MFMA-idle bottleneck (r34: control:mfma ~50%).
Never benched in this campaign (death-list only has pipeb). Single variable = mode.

Routes each shape via recommend_config (same BM/BN/gm/gn for both modes); only mode
differs. Per shape: SNR(pipe), SNR(pipeh) vs fp4_utils ref, then thermal-matched
interleaved hot best-of-3 -> pipeh/pipe perf ratio + geomean. GPU7 only.
"""
import os, sys, math, torch
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


def bench(fn, it=30, wu=8, reps=3):
    fn(); torch.cuda.synchronize()
    for _ in range(wu): fn()
    torch.cuda.synchronize(); best = 1e9
    for _ in range(reps):
        e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(); e0.record()
        for _ in range(it): fn()
        e1.record(); torch.cuda.synchronize(); best = min(best, e0.elapsed_time(e1) * 1000 / it)
    return best


SHAPES = [("7B q/o", 4096, 4096), ("7B gate/up", 11008, 4096), ("7B down", 4096, 11008),
          ("70B q/o", 8192, 8192), ("70B kv", 1024, 8192),
          ("70B gate/up", 28672, 8192), ("70B down", 8192, 28672)]


def main():
    arch = str(get_rocm_arch()); assert "gfx95" in arch, f"needs gfx950, got {arch}"
    dev = "cuda"
    print(f"{'shape':12s}{'M':>6}{'N':>6}{'K':>6}  {'cfg':>16s}  {'SNRpipe':>8s}{'SNRpipeh':>9s}  {'pipe_us':>8s}{'pipeh_us':>9s}  ratio")
    lr = 0.0; n = 0; nwin = 0
    for M in (4096, 8192):
        for tag, N, K in SHAPES:
            BM, BN, gm, gn = recommend_config(M, N, K)
            a, b, asc, bsc = mk(M, N, K, dev)
            ref = torch.matmul(
                fp4_utils.mxfp4_to_f32(a)[:M, :K].float() * fp4_utils.e8m0_to_f32(asc).repeat_interleave(SB, -1)[:M, :K].float(),
                (fp4_utils.mxfp4_to_f32(b)[:N, :K].float() * fp4_utils.e8m0_to_f32(bsc).repeat_interleave(SB, -1)[:N, :K].float()).T)
            sa = ascale(asc, M, K, BM); sb = bscale(bsc, K, BN)
            res = {}
            ok = True
            for mode in ("pipe", "pipeh"):
                try:
                    c = torch.zeros((M, N), dtype=torch.bfloat16, device=dev)
                    fn, mka = build(M, N, K, BM, BN, gm, gn, mode)
                    ar = mka(a, b, c, sa, sb)
                    cc = flyc.compile(fn, *ar); cc(*ar); torch.cuda.synchronize()
                    res[mode] = (snr_db(c, ref), cc, ar)
                except Exception as e:
                    res[mode] = (f"ERR:{type(e).__name__}", None, None); ok = False
            sp = res["pipe"][0]; sh = res["pipeh"][0]
            if not ok or not (isinstance(sp, float) and isinstance(sh, float) and sp >= 40 and sh >= 40):
                print(f"{tag:12s}{M:>6}{N:>6}{K:>6}  {str((BM,BN,gm,gn)):>16s}  {str(sp):>8s}{str(sh):>9s}  (SNR/err -> skip)")
                continue
            # thermal-matched interleaved best-of-3
            bp = min(bench(lambda: res["pipe"][1](*res["pipe"][2])) for _ in range(2))
            bh = min(bench(lambda: res["pipeh"][1](*res["pipeh"][2])) for _ in range(2))
            r = bp / bh  # >1 => pipeh faster
            lr += math.log(r); n += 1; nwin += (r > 1.01)
            print(f"{tag:12s}{M:>6}{N:>6}{K:>6}  {str((BM,BN,gm,gn)):>16s}  {sp:>8.1f}{sh:>9.1f}  {bp*1e3:>8.1f}{bh*1e3:>9.1f}  {r:.3f}")
    if n:
        print(f"\nGEOMEAN pipeh/pipe = {math.exp(lr/n):.4f}  over {n} shapes;  pipeh wins(>1.01): {nwin}/{n}")


if __name__ == "__main__":
    main()
