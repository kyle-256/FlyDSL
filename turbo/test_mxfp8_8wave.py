#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Correctness harness for turbo/mxfp8_gemm_8wave.py (gfx950).

MXFP8: per-1x32 E8M0 block scales on both operands, fed to
v_mfma_scale_f32_16x16x128_f8f6f4. Reference is dequant-then-f32-matmul.

Run:
    python turbo/test_mxfp8_8wave.py [M N K]
"""

import os
import sys

import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_PYFLYDSL_SRC = os.path.join(_REPO_ROOT, "flydsl", "src")
for p in (_REPO_ROOT, _PYFLYDSL_SRC):
    if p not in sys.path:
        sys.path.insert(0, p)

import flydsl.compiler as flyc  # noqa: E402
from flydsl.runtime.device import get_rocm_arch  # noqa: E402
from tests.kernels.utils import fp4_utils  # noqa: E402
from turbo.mxfp8_gemm_8wave import (  # noqa: E402
    compile_mxfp8_gemm_8w,
    preshuffle_scale,
    preshuffle_scale_b_comb,
)

SCALE_BLOCK = 32


def random_fp8_data(rows, cols, device="cuda"):
    # uint8 e4m3 bytes, avoid NaN/Inf encodings (0x7F / 0xFF region)
    return torch.randint(0, 126, (rows, cols), dtype=torch.uint8, device=device)


def random_e8m0(rows, kblocks, lo=125, hi=130, device="cuda"):
    # E8M0 biased exponent bytes; 127 == 2^0. Keep a modest range.
    return torch.randint(lo, hi, (rows, kblocks), dtype=torch.uint8, device=device)


def reference_mxfp8_gemm(a_u8, b_u8, a_scale_u8, b_scale_u8, M, N, K):
    """D = (A*A_scale) @ (B*B_scale)^T in f32. Operands uint8 [M,K]/[N,K],
    scales uint8 E8M0 [M,K//32]/[N,K//32]."""
    a_f32 = fp4_utils.fp8_e4m3_to_f32(a_u8)[:M, :K].float()
    b_f32 = fp4_utils.fp8_e4m3_to_f32(b_u8)[:N, :K].float()
    a_sc = fp4_utils.e8m0_to_f32(a_scale_u8).repeat_interleave(SCALE_BLOCK, dim=-1)[:M, :K].float()
    b_sc = fp4_utils.e8m0_to_f32(b_scale_u8).repeat_interleave(SCALE_BLOCK, dim=-1)[:N, :K].float()
    return torch.matmul(a_f32 * a_sc, (b_f32 * b_sc).T)


def snr_db(out, ref):
    out = out.float()
    ref = ref.float()
    noise = (out - ref).pow(2).mean()
    sig = ref.pow(2).mean()
    if noise.item() == 0:
        return float("inf")
    return (10 * torch.log10(sig / noise)).item()


def run(M=256, N=256, K=512, BLOCK_M=256, BLOCK_N=256):
    arch = str(get_rocm_arch())
    assert "gfx95" in arch, f"needs gfx950, got {arch}"
    assert K % 128 == 0
    dev = "cuda"

    a_u8 = random_fp8_data(M, K, dev)
    b_u8 = random_fp8_data(N, K, dev)
    if os.environ.get("MX_CONST", "0") == "1":
        # all-ones scale (E8M0 127 == 2^0): mxfp8 degenerates to plain fp8 matmul
        a_sc_u8 = torch.full((M, K // SCALE_BLOCK), 127, dtype=torch.uint8, device=dev)
        b_sc_u8 = torch.full((N, K // SCALE_BLOCK), 127, dtype=torch.uint8, device=dev)
    else:
        a_sc_u8 = random_e8m0(M, K // SCALE_BLOCK, device=dev)
        b_sc_u8 = random_e8m0(N, K // SCALE_BLOCK, device=dev)

    ref = reference_mxfp8_gemm(a_u8, b_u8, a_sc_u8, b_sc_u8, M, N, K)

    # pre-shuffle E8M0 (vectorized layout; A fan-out BLOCK_M//64, B BLOCK_N//128)
    a_nt = BLOCK_M // 64
    a_sc_i32 = preshuffle_scale(a_sc_u8, K, a_nt)
    b_sc_i32 = preshuffle_scale_b_comb(b_sc_u8, K)

    c_out = torch.zeros((M, N), dtype=torch.bfloat16, device=dev)

    launch_fn = compile_mxfp8_gemm_8w(K=K, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)

    def _args(c, a, b, sa, sb):
        return (
            a.contiguous().view(torch.int8).view(-1),
            b.contiguous().view(torch.int8).view(-1),
            c.contiguous().view(-1),
            sa.contiguous().view(-1),
            sb.contiguous().view(-1),
            M,
            N,
            torch.cuda.current_stream(),
        )

    compiled = flyc.compile(launch_fn, *_args(c_out, a_u8, b_u8, a_sc_i32, b_sc_i32))
    compiled(*_args(c_out, a_u8, b_u8, a_sc_i32, b_sc_i32))
    torch.cuda.synchronize()

    out = c_out.float()
    s = snr_db(out, ref)
    print(f"[mxfp8_8wave] M={M} N={N} K={K}  SNR={s:.2f} dB")
    print(f"  ref[:2,:4]=\n{ref[:2,:4]}")
    print(f"  out[:2,:4]=\n{out[:2,:4]}")
    ok = s > 20.0
    print("  RESULT:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    args = [int(x) for x in sys.argv[1:]]
    M, N, K = (args + [256, 256, 512])[:3]
    run(M, N, K)
