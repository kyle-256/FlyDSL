"""round-54 (Mode-B B10 r10_1, pipe leg): BLOCK_M=128 kv on PRODUCTION pipe.

staged BM128/BN128 already PASS (SNR 55.62 + DET0). The pre-campaign death-list
said "BM128 races on the fp4 PIPE" specifically -> the decisive test is the pipe
path, which has the SW-pipeline in-place LDS recycle + wait_barrier(1) (r_k7) +
narrow-A-G2S (r6_3 ported it for BM192; BM128 has A_NARROW=True identically).

Stronger det than the staged 5-run: 2 fresh-data passes x 200 runs bitwise.
The pre-campaign symptom was "SNR -3dB + nondeterministic" so both SNR and
det0 must hold. PASS = pipe BM128 race-free -> B10 GO (r10_2 = route+perf).
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


def compile_pipe(K, BM, BN):
    return compile_mxfp4_gemm_8w(K=K, BLOCK_M=BM, BLOCK_N=BN, mode="pipe")


def one(fn, a, b, asp, bsp, M, N):
    d = "cuda"
    st = torch.cuda.current_stream()
    c = torch.zeros((M, N), dtype=torch.bfloat16, device=d)
    args = (a.view(torch.int8).view(-1), b.view(torch.int8).view(-1), c.view(-1),
            asp.view(-1), bsp.view(-1), M, N, st)
    cc = flyc.compile(fn, *args); cc(*args); torch.cuda.synchronize()
    return c


for M, N, K in [(4096, 1024, 8192), (8192, 1024, 8192)]:
    try:
        fn = compile_pipe(K, 128, 128)
        worst_md = 0.0
        snr0 = None
        for p in range(2):  # fresh data per pass (race is data-dependent)
            a, b, asc, bsc = build(M, N, K)
            ref = ref_mxfp4(a, b, asc, bsc, M, N, K)
            asp = preshuffle_scale(asc, K, 128 // 64)
            bsp = preshuffle_scale(bsc, K, 128 // 128)
            base = one(fn, a, b, asp, bsp, M, N)
            if p == 0:
                snr0 = snr(base.float(), ref)
            for _ in range(200):
                c = one(fn, a, b, asp, bsp, M, N)
                md = (c.float() - base.float()).abs().max().item()
                worst_md = max(worst_md, md)
                if md != 0.0:
                    break
        det = "DET0" if worst_md == 0.0 else f"NONDET(maxdiff={worst_md:.4g})"
        ok = "PASS" if (snr0 >= 40.0 and worst_md == 0.0) else "FAIL"
        print(f"kv M={M} N={N} K={K} [PIPE]: BM128/BN128 SNR={snr0:6.2f} "
              f"{det} (2pass x 200run fresh) -> {ok}")
    except Exception as e:
        print(f"kv M={M} N={N} K={K} [PIPE]: BM128 ERROR: {type(e).__name__}: {str(e)[:200]}")
