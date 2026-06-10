"""round-57 (Mode-B B11 r_a1): combined-128 A G2S for BM128, STAGED correctness.

The BM128 narrow-A-G2S clamp-wave runs only NW_A_ACTIVE=4 of 8 waves usefully
(waves 4-7 redundantly reload -> 50% A VMEM waste + 2x A-G2S issues). B11 mirrors
the proven BN128 B combined-128 trick: ONE 128-row G2S over LDS-adjacent
a_lds0+a_lds1 fills both halves with all 8 waves. This round = STAGED correctness
vehicle (one fill/K-iter, no pipeline): does combined-A produce correct SNR + det0?
PASS -> r_a2 ports to production pipe (with merged-spill wait tuning like B r_k7).
"""
import os, sys, torch
_R = "/workspace/code/FlyDSL"
for p in (_R, _R + "/flydsl/src"):
    sys.path.insert(0, p)
import flydsl.compiler as flyc
from tests.kernels.utils import fp4_utils
from turbo.mxfp4_gemm_8wave import compile_mxfp4_gemm_8w
from turbo.mxfp8_gemm_8wave import preshuffle_scale
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


def build(M, N, K):
    d = "cuda"
    a = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device=d)
    b = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=d)
    asc = torch.randint(125, 130, (M, K // SB), dtype=torch.uint8, device=d)
    bsc = torch.randint(125, 130, (N, K // SB), dtype=torch.uint8, device=d)
    return a, b, asc, bsc


def run(a, b, asc, bsc, M, N, K, BM, BN, det_runs=1):
    asp = preshuffle_scale(asc, K, BM // 64)
    bsp = preshuffle_scale(bsc, K, BN // 128)
    st = torch.cuda.current_stream()
    fn = compile_mxfp4_gemm_8w(K=K, BLOCK_M=BM, BLOCK_N=BN, mode="staged")
    outs = []
    for _ in range(det_runs):
        c = torch.zeros((M, N), dtype=torch.bfloat16, device="cuda")
        args = (a.view(torch.int8).view(-1), b.view(torch.int8).view(-1), c.view(-1),
                asp.view(-1), bsp.view(-1), M, N, st)
        cc = flyc.compile(fn, *args); cc(*args); torch.cuda.synchronize()
        outs.append(c.clone())
    return outs


for M, N, K in [(4096, 1024, 8192), (8192, 1024, 8192)]:
    a, b, asc, bsc = build(M, N, K)
    ref = ref_mxfp4(a, b, asc, bsc, M, N, K)
    o256 = run(a, b, asc, bsc, M, N, K, 256, 128)[0]
    s256 = snr(o256.float(), ref)
    try:
        outs = run(a, b, asc, bsc, M, N, K, 128, 128, det_runs=5)
        s128 = snr(outs[0].float(), ref)
        md = max((o.float() - outs[0].float()).abs().max().item() for o in outs[1:])
        det = "DET0" if md == 0.0 else f"NONDET({md:.4g})"
        ok = "PASS" if (s128 >= 40.0 and md == 0.0) else "FAIL"
        print(f"kv M={M} [staged combineA]: BM256/BN128 SNR={s256:6.2f} | "
              f"BM128/BN128(combineA) SNR={s128:6.2f} {det} -> {ok}")
    except Exception as e:
        print(f"kv M={M} [staged combineA]: BM256 SNR={s256:6.2f} | BM128 ERROR: {type(e).__name__}: {str(e)[:160]}")
