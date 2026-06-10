"""r_k3 smoke: BLOCK_N=128 combined-128 B G2S in STAGED mode on kv.

Verifies the G2S fix: BN128 B half-regions are < _ROWS_PER_STEP(128), so the
two adjacent 64-row halves are filled by ONE 128-row G2S (rows 0..63->b_lds0,
64..127->b_lds1). Also exercises the per-region scale gate (load_halves). vs
dequant f32 reference. PASS if BN128 SNR >= 40 dB; BN256 staged anchor printed
as a sanity check (staged is not production, need not be bit-identical)."""
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
    fn = compile_mxfp4_gemm_8w(K=K, BLOCK_M=256, BLOCK_N=BN, mode="staged")
    args = (a.view(torch.int8).view(-1), b.view(torch.int8).view(-1), c.view(-1),
            asp.view(-1), bsp.view(-1), M, N, st)
    cc = flyc.compile(fn, *args); cc(*args); torch.cuda.synchronize()
    return snr(c.float(), ref)


for M, N, K in [(4096, 1024, 8192), (8192, 1024, 8192)]:
    s256 = run(M, N, K, 256)
    try:
        s128 = run(M, N, K, 128)
        ok = "PASS" if s128 >= 40.0 else "FAIL"
        print(f"kv M={M} N={N} K={K} [staged]: BN256 SNR={s256:6.2f}dB | BN128 SNR={s128:6.2f}dB -> {ok}")
    except Exception as e:
        print(f"kv M={M} N={N} K={K} [staged]: BN256 SNR={s256:6.2f}dB | BN128 ERROR: {type(e).__name__}: {str(e)[:140]}")
