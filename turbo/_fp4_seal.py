"""Round-39 production-routing correctness seal for turbo/mxfp4_gemm_8wave.py.

UNLIKE test_/det_mxfp4_8wave.py (fixed BM256/BN256), this routes every one of the
14 Llama production shapes through recommend_config (the ACTUAL production path:
kv -> BN128/gm2/BM192, big-N -> gn14, bulk -> BN256/gm4) and validates:
  (1) SNR >= 40 dB vs fp4_utils dequant reference  on all 14 routed shapes
  (2) det0 (>=300-run, fresh data each pass)        on each DISTINCT routed config class

Measurement-only. No kernel change. mode='pipe' (production).
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
    nta = BM // 64
    q = 16 * nta
    pad = ((M + q - 1) // q) * q
    if pad != M:
        ap = torch.zeros((pad, K // SB), dtype=torch.uint8, device=asc.device)
        ap[:M] = asc
        asc = ap
    return preshuffle_scale(asc, K, nta).view(-1)


def bscale(bsc, K, BN):
    return (preshuffle_scale_b_comb(bsc, K) if BN >= 256 else preshuffle_scale(bsc, K, BN // 128)).view(-1)


def snr_db(out, ref):
    out, ref = out.float(), ref.float()
    noise = (out - ref).pow(2).mean(); sig = ref.pow(2).mean()
    return float("inf") if noise.item() == 0 else (10 * torch.log10(sig / noise)).item()


def mk(M, N, K, dev="cuda"):
    a = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device=dev)
    b = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=dev)
    asc = torch.randint(124, 131, (M, K // SB), dtype=torch.uint8, device=dev)
    bsc = torch.randint(124, 131, (N, K // SB), dtype=torch.uint8, device=dev)
    return a, b, asc, bsc


def build(M, N, K, BM, BN, gm, gn, nx=8):
    fn = compile_mxfp4_gemm_8w(K=K, BLOCK_M=BM, BLOCK_N=BN, mode="pipe", group_m=gm, group_n=gn, num_xcds=nx)
    def args(a, b, c, sa, sb):
        return (a.view(torch.int8).view(-1), b.view(torch.int8).view(-1), c.view(-1),
                sa, sb, M, N, torch.cuda.current_stream())
    return fn, args


SHAPES = [("7B q/o", 4096, 4096), ("7B gate/up", 11008, 4096), ("7B down", 4096, 11008),
          ("70B q/o", 8192, 8192), ("70B kv", 1024, 8192),
          ("70B gate/up", 28672, 8192), ("70B down", 8192, 28672)]


def main():
    arch = str(get_rocm_arch()); assert "gfx95" in arch, f"needs gfx950, got {arch}"
    dev = "cuda"
    print("=== Part 1: SNR vs fp4_utils reference (recommend_config-routed, mode=pipe) ===")
    print(f"{'shape':12s}{'M':>6}{'N':>6}{'K':>6}  {'cfg(BM,BN,gm,gn)':>18s}  {'SNR(dB)':>8s}  res")
    worst_snr = 1e9; n_snr_pass = 0; classes = {}
    for M in (4096, 8192):
        for tag, N, K in SHAPES:
            BM, BN, gm, gn, nx = recommend_config(M, N, K)
            a, b, asc, bsc = mk(M, N, K, dev)
            ref = torch.matmul(
                fp4_utils.mxfp4_to_f32(a)[:M, :K].float() * fp4_utils.e8m0_to_f32(asc).repeat_interleave(SB, -1)[:M, :K].float(),
                (fp4_utils.mxfp4_to_f32(b)[:N, :K].float() * fp4_utils.e8m0_to_f32(bsc).repeat_interleave(SB, -1)[:N, :K].float()).T)
            sa = ascale(asc, M, K, BM); sb = bscale(bsc, K, BN)
            c = torch.zeros((M, N), dtype=torch.bfloat16, device=dev)
            fn, mkargs = build(M, N, K, BM, BN, gm, gn, nx)
            ar = mkargs(a, b, c, sa, sb)
            cc = flyc.compile(fn, *ar); cc(*ar); torch.cuda.synchronize()
            s = snr_db(c, ref)
            ok = s >= 40.0; n_snr_pass += ok; worst_snr = min(worst_snr, s)
            cls = (BM, BN, gm, gn, nx)
            classes.setdefault(cls, (tag, M, N, K))
            print(f"{tag:12s}{M:>6}{N:>6}{K:>6}  {str(cls):>18s}  {s:>8.2f}  {'PASS' if ok else 'FAIL'}")
    print(f"\nSNR: {n_snr_pass}/14 >= 40 dB, worst = {worst_snr:.2f} dB")

    print("\n=== Part 2: det0 (300-run fresh-data) per DISTINCT routed config class ===")
    runs = int(os.environ.get("SEAL_RUNS", "300"))
    print(f"{'class(BM,BN,gm,gn)':>20s}  {'rep-shape':12s}{'M':>6}{'N':>6}{'K':>6}  {'max_diff':>10s}  res")
    all_det0 = True
    for cls, (tag, M, N, K) in sorted(classes.items()):
        BM, BN, gm, gn, nx = cls
        fn, mkargs = build(M, N, K, BM, BN, gm, gn, nx)
        worst = 0.0; first_bad = -1
        for pi in range(3):
            a, b, asc, bsc = mk(M, N, K, dev)
            sa = ascale(asc, M, K, BM); sb = bscale(bsc, K, BN)
            c = torch.zeros((M, N), dtype=torch.bfloat16, device=dev)
            ar = mkargs(a, b, c, sa, sb)
            cc = flyc.compile(fn, *ar); cc(*ar); torch.cuda.synchronize()
            refc = c.clone()
            for ri in range(runs):
                c.zero_(); cc(*ar)
                d = (c.float() - refc.float()).abs().max().item()
                if d > worst: worst = d
                if d > 0: first_bad = ri; break
            if first_bad >= 0: break
        ok = worst == 0.0; all_det0 = all_det0 and ok
        print(f"{str(cls):>20s}  {tag:12s}{M:>6}{N:>6}{K:>6}  {worst:>10.2e}  {'DET0' if ok else 'RACE'}")
    print(f"\nOVERALL: SNR {n_snr_pass}/14 pass + det {'ALL DET0' if all_det0 else 'RACE DETECTED'}")
    print("SEAL:", "PASS" if (n_snr_pass == 14 and all_det0) else "FAIL")


if __name__ == "__main__":
    main()
