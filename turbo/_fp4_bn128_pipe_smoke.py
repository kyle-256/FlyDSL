"""r_k2 smoke: BLOCK_N=128 in the PRODUCTION pipe kernel (mode="pipe") on kv.

Verifies (1) BN256 pipe stays bit-identical (anchor unchanged by the load_halves
refactor) and (2) whether BN128 pipe is correct. Per-region B-scale via the new
load_halves path; host b-scale = preshuffle_scale(b_sc, K, N_TILES_B). vs dequant
f32 reference. PASS if BN128 SNR >= 40 dB."""
import os, sys, torch
_R = "/workspace/code/FlyDSL"
for p in (_R, _R + "/flydsl/src"):
    sys.path.insert(0, p)
import flydsl.compiler as flyc
from tests.kernels.utils import fp4_utils
from turbo.mxfp4_gemm_8wave import compile_mxfp4_gemm_8w
from turbo.mxfp8_gemm_8wave import preshuffle_scale, preshuffle_scale_b_comb
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


def run(M, N, K, BN):
    d = "cuda"
    a = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device=d)
    b = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=d)
    asc = torch.randint(125, 130, (M, K // SB), dtype=torch.uint8, device=d)
    bsc = torch.randint(125, 130, (N, K // SB), dtype=torch.uint8, device=d)
    ref = ref_mxfp4(a, b, asc, bsc, M, N, K)
    asp = preshuffle_scale(asc, K, 256 // 64)
    bsp = preshuffle_scale_b_comb(bsc, K) if BN >= 256 else preshuffle_scale(bsc, K, BN // 128)
    c = torch.zeros((M, N), dtype=torch.bfloat16, device=d)
    st = torch.cuda.current_stream()
    fn = compile_mxfp4_gemm_8w(K=K, BLOCK_M=256, BLOCK_N=BN, mode="pipe")
    args = (a.view(torch.int8).view(-1), b.view(torch.int8).view(-1), c.view(-1),
            asp.view(-1), bsp.view(-1), M, N, st)
    cc = flyc.compile(fn, *args); cc(*args); torch.cuda.synchronize()
    return snr(c.float(), ref)


for M, N, K in [(4096, 1024, 8192), (8192, 1024, 8192)]:
    s256 = run(M, N, K, 256)
    try:
        s128 = run(M, N, K, 128)
        ok = "PASS" if s128 >= 40.0 else "FAIL"
        print(f"kv M={M} N={N} K={K} [pipe]: BN256(anchor) SNR={s256:6.2f}dB | BN128 SNR={s128:6.2f}dB -> {ok}")
    except Exception as e:
        print(f"kv M={M} N={N} K={K} [pipe]: BN256(anchor) SNR={s256:6.2f}dB | BN128 ERROR: {type(e).__name__}: {str(e)[:120]}")
