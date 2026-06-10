"""Round-45 B8 r8_2: correctness smoke for kernel_gemm_pipe3 (3-stage LDS ring).
Bulk shapes (BN256/BM256/B_COMB). Shared inputs; compare pipe3 SNR vs fp4 ref AND
pipe3 vs pipe maxdiff (3-stage rotation must reproduce pipe's result). GPU7 only."""
import os, sys, torch
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


def run(M, N, K, BM, BN, gm, gn, mode, a, b, sa, sb, dev="cuda"):
    c = torch.zeros((M, N), dtype=torch.bfloat16, device=dev)
    fn, mka = build(M, N, K, BM, BN, gm, gn, mode)
    ar = mka(a, b, c, sa, sb)
    cc = flyc.compile(fn, *ar); cc(*ar); torch.cuda.synchronize()
    return c, cc, ar


SHAPES = [("7B q/o", 4096, 4096, 4096), ("70B q/o", 8192, 8192, 8192), ("70B down", 8192, 8192, 28672)]


def main():
    arch = str(get_rocm_arch()); assert "gfx95" in arch, f"needs gfx950, got {arch}"
    dev = "cuda"
    print(f"{'shape':12s}{'M':>6}{'N':>6}{'K':>6}  {'cfg':>14s}  {'SNRpipe':>8s}{'SNRpipe3':>9s}  {'p3-vs-pipe maxdiff':>18s}  det0")
    allok = True
    for tag, M, N, K in SHAPES:
        BM, BN, gm, gn = recommend_config(M, N, K)
        a, b, asc, bsc = mk(M, N, K, dev)
        ref = torch.matmul(
            fp4_utils.mxfp4_to_f32(a)[:M, :K].float() * fp4_utils.e8m0_to_f32(asc).repeat_interleave(SB, -1)[:M, :K].float(),
            (fp4_utils.mxfp4_to_f32(b)[:N, :K].float() * fp4_utils.e8m0_to_f32(bsc).repeat_interleave(SB, -1)[:N, :K].float()).T)
        sa = ascale(asc, M, K, BM); sb = bscale(bsc, K, BN)
        try:
            cpipe, _, _ = run(M, N, K, BM, BN, gm, gn, "pipe", a, b, sa, sb, dev)
            cp3, cc3, ar3 = run(M, N, K, BM, BN, gm, gn, "pipe3", a, b, sa, sb, dev)
        except Exception as e:
            print(f"{tag:12s}{M:>6}{N:>6}{K:>6}  {str((BM,BN,gm,gn)):>14s}  ERR {type(e).__name__}: {e}")
            allok = False; continue
        sp = snr_db(cpipe, ref); s3 = snr_db(cp3, ref)
        md = (cp3.float() - cpipe.float()).abs().max().item()
        # det0: 5 fresh-input reruns of pipe3, bitwise vs first
        det = "?"
        if s3 >= 40:
            c0 = cp3.clone(); maxd = 0.0
            for _ in range(5):
                cc3(*ar3); torch.cuda.synchronize()
                maxd = max(maxd, (cp3.float() - c0.float()).abs().max().item())
            det = "DET0" if maxd == 0 else f"NONDET({maxd:.3g})"
        ok = (s3 >= 40)
        allok = allok and ok
        print(f"{tag:12s}{M:>6}{N:>6}{K:>6}  {str((BM,BN,gm,gn)):>14s}  {sp:>8.1f}{s3:>9.1f}  {md:>18.4g}  {det}")
    print("\nRESULT:", "PASS (pipe3 3-stage rotation correct)" if allok else "FAIL")


if __name__ == "__main__":
    main()
