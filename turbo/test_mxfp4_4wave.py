#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Correctness harness for turbo/mxfp4_gemm_4wave.py (gfx950, 4-wave 2x2).

Scales packed with preshuffle_scale(., K, n_tiles) for BOTH A and B,
n_tiles = BLOCK//4//16. Run: python turbo/test_mxfp4_4wave.py [M N K]
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
from turbo.mxfp4_gemm_4wave import compile_mxfp4_gemm_4w  # noqa: E402
from turbo.mxfp8_gemm_8wave import preshuffle_scale  # noqa: E402

SCALE_BLOCK = 32


def reference_mxfp4_gemm(a_u8, b_u8, a_sc_u8, b_sc_u8, M, N, K):
    a_f = fp4_utils.mxfp4_to_f32(a_u8)[:M, :K].float()
    b_f = fp4_utils.mxfp4_to_f32(b_u8)[:N, :K].float()
    a_s = fp4_utils.e8m0_to_f32(a_sc_u8).repeat_interleave(SCALE_BLOCK, dim=-1)[:M, :K].float()
    b_s = fp4_utils.e8m0_to_f32(b_sc_u8).repeat_interleave(SCALE_BLOCK, dim=-1)[:N, :K].float()
    return torch.matmul(a_f * a_s, (b_f * b_s).T)


def snr_db(out, ref):
    out, ref = out.float(), ref.float()
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

    a_u8 = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device=dev)
    b_u8 = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=dev)
    a_sc_u8 = torch.randint(125, 130, (M, K // SCALE_BLOCK), dtype=torch.uint8, device=dev)
    b_sc_u8 = torch.randint(125, 130, (N, K // SCALE_BLOCK), dtype=torch.uint8, device=dev)

    ref = reference_mxfp4_gemm(a_u8, b_u8, a_sc_u8, b_sc_u8, M, N, K)

    nta = BLOCK_M // 4 // 16
    ntb = BLOCK_N // 4 // 16
    a_sc_i32 = preshuffle_scale(a_sc_u8, K, nta)
    b_sc_i32 = preshuffle_scale(b_sc_u8, K, ntb)

    pad = int(os.environ.get("FP4_PAD", "0"))
    asm_mfma = int(os.environ.get("FP4_ASM", "0")) > 0
    il = int(os.environ.get("FP4_IL", "0")) > 0
    se = int(os.environ.get("FP4_SE", "0")) > 0
    c_out = torch.zeros((M, N), dtype=torch.bfloat16, device=dev)
    launch_fn = compile_mxfp4_gemm_4w(
        K=K, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, padded=pad > 0, pad_bytes=max(pad, 16),
        asm_mfma=asm_mfma, interleave=il, asm_se=se,
    )

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

    s = snr_db(c_out.float(), ref)
    print(f"[mxfp4_4wave pad={pad}] M={M} N={N} K={K}  SNR={s:.2f} dB")
    print(f"  ref[:2,:4]=\n{ref[:2, :4]}")
    print(f"  out[:2,:4]=\n{c_out.float()[:2, :4]}")
    print("  RESULT:", "PASS" if s > 20.0 else "FAIL")
    return s > 20.0


if __name__ == "__main__":
    args = [int(x) for x in sys.argv[1:]]
    M, N, K = (args + [256, 256, 512])[:3]
    run(M, N, K)
