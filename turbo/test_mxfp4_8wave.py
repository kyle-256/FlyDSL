#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Correctness harness for turbo/mxfp4_gemm_8wave.py (gfx950).

MXFP4: E2M1 fp4 packed 2-per-byte, per-1x32 E8M0 block scales on both operands,
fed to v_mfma_scale_f32_16x16x128_f8f6f4 (cbsz=4/blgp=4 fp4 mode). Reference is
dequant (mxfp4_to_f32 + e8m0) then f32 matmul.

Run:  python turbo/test_mxfp4_8wave.py [M N K]    (K = logical fp4 count)
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
from turbo.mxfp4_gemm_8wave import compile_mxfp4_gemm_8w, preshuffle_mxfp4_scales  # noqa: E402
from turbo.mxfp8_gemm_8wave import preshuffle_scale, preshuffle_scale_b_comb  # noqa: E402

SCALE_BLOCK = 32


def random_fp4_packed(rows, k_logical, device="cuda"):
    """Random packed MXFP4: uint8 [rows, k_logical//2], 2 fp4 nibbles per byte.
    fp4 codes 0..7 / 8..15 map to {0,.5,1,1.5,2,3,4,6} and negatives; avoid no special NaN."""
    return torch.randint(0, 256, (rows, k_logical // 2), dtype=torch.uint8, device=device)


def random_e8m0(rows, kblocks, lo=125, hi=130, device="cuda"):
    return torch.randint(lo, hi, (rows, kblocks), dtype=torch.uint8, device=device)


def reference_mxfp4_gemm(a_u8, b_u8, a_sc_u8, b_sc_u8, M, N, K):
    """D = (A*A_scale) @ (B*B_scale)^T. a/b packed fp4 [.,K/2], scales E8M0 [.,K/32]."""
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

    a_u8 = random_fp4_packed(M, K, dev)  # [M, K/2]
    b_u8 = random_fp4_packed(N, K, dev)  # [N, K/2]
    a_sc_u8 = random_e8m0(M, K // SCALE_BLOCK, device=dev)
    b_sc_u8 = random_e8m0(N, K // SCALE_BLOCK, device=dev)

    ref = reference_mxfp4_gemm(a_u8, b_u8, a_sc_u8, b_sc_u8, M, N, K)

    mode = os.environ.get("FP4_MODE", "direct")
    block_k = int(os.environ.get("FP4_BLOCK_K", "128"))
    pad = int(os.environ.get("FP4_PAD", "0"))
    asm_mfma = int(os.environ.get("FP4_ASM", "0")) > 0
    # Host scale prep matches the kernel's chosen layout (packed for eligible pipe).
    a_sc_i32, b_sc_i32 = preshuffle_mxfp4_scales(
        a_sc_u8, b_sc_u8, K, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, mode=mode, asm_mfma=asm_mfma, padded=pad > 0)
    c_out = torch.zeros((M, N), dtype=torch.bfloat16, device=dev)
    launch_fn = compile_mxfp4_gemm_8w(
        K=K, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, mode=mode, block_k=block_k, padded=pad > 0, pad_bytes=max(pad, 16),
        asm_mfma=asm_mfma, asm_se=int(os.environ.get("FP4_SE", "0")) > 0, frag_pad=int(os.environ.get("FP4_NOPAD","0"))==0, sched=int(os.environ.get("FP4_SCHED","0"))>0, iglp=int(os.environ.get("FP4_IGLP","0"))>0,
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
    print(f"[mxfp4_8wave mode={mode}] M={M} N={N} K={K}  SNR={s:.2f} dB")
    print(f"  ref[:2,:4]=\n{ref[:2, :4]}")
    print(f"  out[:2,:4]=\n{c_out.float()[:2, :4]}")
    print("  RESULT:", "PASS" if s > 20.0 else "FAIL")
    return s > 20.0


if __name__ == "__main__":
    args = [int(x) for x in sys.argv[1:]]
    M, N, K = (args + [256, 256, 512])[:3]
    run(M, N, K)
