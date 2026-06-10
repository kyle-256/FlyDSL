"""round-54 (Mode-B B10 r10_1): BLOCK_M=128 kv feasibility w/ narrow-A-G2S.

Hypothesis: the pre-campaign "BLOCK_M=128 races (SNR -3dB + nondeterministic)"
death-list entry PRE-DATES the campaign's two race-fix mechanisms — narrow-A-G2S
clamp-wave (r6_2/r6_3, built for BM192) and wait_barrier(1) (r_k7, fixed the
BN128 combined-G2S race). BM128 is structurally expressible (assert passes,
LDS_BLOCK_M=64<128 -> A_NARROW path, N_TILES_A=2, NW_A_ACTIVE=4) and reuses the
SAME narrow-A-G2S machinery as the landed BM192. This is the "全新机制" exception
to the death-list. BM128 on kv M4096 = 32 M-tiles x 8 N-tiles = 256wg (FULL) vs
BM192's 176wg -> directly attacks the binding min_ratio (kv M4096 = 0.567).

This round = STAGED correctness vehicle (mirrors r6_2/r8_2): does staged
BM128/BN128 produce correct SNR AND det0 on kv? Staged has no SW-pipeline
in-place LDS recycle, so it isolates the BM128 topology/G2S/scale structure from
the pipe race. PASS here = structure sound -> r10_2 pipe port + wait(1). FAIL =
race is structural (N_TILES_A=2 + LDS_BLOCK_M=64 layout), B10 harder.
"""
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


def build(M, N, K):
    d = "cuda"
    a = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device=d)
    b = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=d)
    asc = torch.randint(125, 130, (M, K // SB), dtype=torch.uint8, device=d)
    bsc = torch.randint(125, 130, (N, K // SB), dtype=torch.uint8, device=d)
    return a, b, asc, bsc


def run(a, b, asc, bsc, M, N, K, BM, BN, det_runs=1):
    d = "cuda"
    asp = preshuffle_scale(asc, K, BM // 64)
    bsp = preshuffle_scale_b_comb(bsc, K) if BN >= 256 else preshuffle_scale(bsc, K, BN // 128)
    st = torch.cuda.current_stream()
    fn = compile_mxfp4_gemm_8w(K=K, BLOCK_M=BM, BLOCK_N=BN, mode="staged")
    outs = []
    for _ in range(det_runs):
        c = torch.zeros((M, N), dtype=torch.bfloat16, device=d)
        args = (a.view(torch.int8).view(-1), b.view(torch.int8).view(-1), c.view(-1),
                asp.view(-1), bsp.view(-1), M, N, st)
        cc = flyc.compile(fn, *args); cc(*args); torch.cuda.synchronize()
        outs.append(c.clone())
    return outs


for M, N, K in [(4096, 1024, 8192), (8192, 1024, 8192)]:
    a, b, asc, bsc = build(M, N, K)
    ref = ref_mxfp4(a, b, asc, bsc, M, N, K)
    # anchor: known-good BM256/BN128
    o256 = run(a, b, asc, bsc, M, N, K, 256, 128)[0]
    s256 = snr(o256.float(), ref)
    try:
        # BM128/BN128 — feasibility + det0 (5 runs same input, bitwise)
        outs = run(a, b, asc, bsc, M, N, K, 128, 128, det_runs=5)
        s128 = snr(outs[0].float(), ref)
        maxdiff = 0.0
        for o in outs[1:]:
            md = (o.float() - outs[0].float()).abs().max().item()
            maxdiff = max(maxdiff, md)
        det = "DET0" if maxdiff == 0.0 else f"NONDET(maxdiff={maxdiff:.4g})"
        ok = "PASS" if (s128 >= 40.0 and maxdiff == 0.0) else "FAIL"
        print(f"kv M={M} N={N} K={K} [staged]: BM256/BN128 SNR={s256:6.2f} | "
              f"BM128/BN128 SNR={s128:6.2f} {det} -> {ok}")
    except Exception as e:
        print(f"kv M={M} N={N} K={K} [staged]: BM256/BN128 SNR={s256:6.2f} | "
              f"BM128 ERROR: {type(e).__name__}: {str(e)[:160]}")
