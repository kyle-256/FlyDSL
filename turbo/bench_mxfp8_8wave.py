#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Perf harness for turbo/mxfp8_gemm_8wave.py on Llama 7B / 70B GEMM shapes.

GEMM convention: C[M,N] = A[M,K] @ B_T[N,K]^T   (N = out features, K = in features)

Run:
    python turbo/bench_mxfp8_8wave.py            # default M sweep
    python turbo/bench_mxfp8_8wave.py 4096        # fixed M
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
from kernels.fp8_gemm_8wave import compile_fp8_gemm_8w  # noqa: E402
from tests.kernels.utils import fp4_utils  # noqa: E402
from turbo.mxfp8_gemm_8wave import compile_mxfp8_gemm_8w, preshuffle_scale, preshuffle_scale_b_comb  # noqa: E402

SCALE_BLOCK = 32


def bench(fn, it=50, warmup=10, reps=3):
    """Hot-L2 best-of-reps timing (matches scripts2/_h.py). Returns median-best us."""
    fn()
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(reps):
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        e0.record()
        for _ in range(it):
            fn()
        e1.record()
        torch.cuda.synchronize()
        best = min(best, e0.elapsed_time(e1) * 1000.0 / it)
    return best

# (label, N=out, K=in)
LLAMA_7B = [
    ("7B  q/k/v/o   ", 4096, 4096),
    ("7B  gate/up   ", 11008, 4096),
    ("7B  down      ", 4096, 11008),
]
LLAMA_70B = [
    ("70B q/o       ", 8192, 8192),
    ("70B kv (gqa)  ", 1024, 8192),
    ("70B gate/up   ", 28672, 8192),
    ("70B down      ", 8192, 28672),
]


def _mk(M, N, K, dev="cuda"):
    a_u8 = torch.randint(0, 126, (M, K), dtype=torch.uint8, device=dev)
    b_u8 = torch.randint(0, 126, (N, K), dtype=torch.uint8, device=dev)
    a_sc = torch.randint(125, 130, (M, K // SCALE_BLOCK), dtype=torch.uint8, device=dev)
    b_sc = torch.randint(125, 130, (N, K // SCALE_BLOCK), dtype=torch.uint8, device=dev)
    return a_u8, b_u8, a_sc, b_sc


def _ref(a_u8, b_u8, a_sc, b_sc, M, N, K):
    a_f = fp4_utils.fp8_e4m3_to_f32(a_u8)[:M, :K].float()
    b_f = fp4_utils.fp8_e4m3_to_f32(b_u8)[:N, :K].float()
    asc = fp4_utils.e8m0_to_f32(a_sc).repeat_interleave(SCALE_BLOCK, dim=-1)[:M, :K].float()
    bsc = fp4_utils.e8m0_to_f32(b_sc).repeat_interleave(SCALE_BLOCK, dim=-1)[:N, :K].float()
    return torch.matmul(a_f * asc, (b_f * bsc).T)


def _snr(out, ref):
    out, ref = out.float(), ref.float()
    n = (out - ref).pow(2).mean()
    s = ref.pow(2).mean()
    return float("inf") if n.item() == 0 else (10 * torch.log10(s / n)).item()


def bench_one(M, N, K, *, BLOCK_M=256, BLOCK_N=256, iters=20, warmup=5, check=True):
    dev = "cuda"
    a_u8, b_u8, a_sc, b_sc = _mk(M, N, K, dev)
    a_sc_i32 = preshuffle_scale(a_sc, K, BLOCK_M // 64)
    b_sc_i32 = preshuffle_scale_b_comb(b_sc, K)
    c_out = torch.zeros((M, N), dtype=torch.bfloat16, device=dev)

    launch_fn = compile_mxfp8_gemm_8w(K=K, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)

    def _args(c):
        return (
            a_u8.view(torch.int8).view(-1),
            b_u8.view(torch.int8).view(-1),
            c.view(-1),
            a_sc_i32.view(-1),
            b_sc_i32.view(-1),
            M,
            N,
            torch.cuda.current_stream(),
        )

    compiled = flyc.compile(launch_fn, *_args(c_out))
    args = _args(c_out)
    us = bench(lambda: compiled(*args))

    tflops = 2 * M * N * K / (us / 1e6) / 1e12
    bytes_moved = M * K + N * K + M * N * 2 + (M + N) * (K // SCALE_BLOCK)
    tbps = bytes_moved / 1e12 / (us / 1e6)

    snr = _snr(c_out.float(), _ref(a_u8, b_u8, a_sc, b_sc, M, N, K)) if check else None
    return us, tflops, tbps, snr


def bench_tensorwise(M, N, K, *, BLOCK_M=256, BLOCK_N=256, iters=20, warmup=5):
    """Reference: the tensorwise fp8_gemm_8wave (same kernel family, no per-block scale)."""
    dev = "cuda"
    a_u8 = torch.randint(0, 126, (M, K), dtype=torch.uint8, device=dev)
    b_u8 = torch.randint(0, 126, (N, K), dtype=torch.uint8, device=dev)
    sa = torch.ones(M, dtype=torch.float32, device=dev)
    sb = torch.ones(N, dtype=torch.float32, device=dev)
    c_out = torch.zeros((M, N), dtype=torch.bfloat16, device=dev)
    launch_fn = compile_fp8_gemm_8w(K=K, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)

    def _args(c):
        return (
            a_u8.view(torch.int8).view(-1),
            b_u8.view(torch.int8).view(-1),
            c.view(-1),
            sa.view(-1),
            sb.view(-1),
            M,
            N,
            torch.cuda.current_stream(),
        )

    compiled = flyc.compile(launch_fn, *_args(c_out))
    args = _args(c_out)
    us = bench(lambda: compiled(*args))
    return us, 2 * M * N * K / (us / 1e6) / 1e12


def main():
    arch = str(get_rocm_arch())
    assert "gfx95" in arch, f"needs gfx950, got {arch}"
    argv = [int(x) for x in sys.argv[1:]]
    Ms = argv if argv else [2048, 4096, 8192]

    for M in Ms:
        print(f"\n===== M = {M} =====")
        print(
            f"{'shape':<16}{'N':>7}{'K':>7}"
            f"{'mx_us':>9}{'mx_TF':>8}{'tw_us':>9}{'tw_TF':>8}{'mx/tw':>7}{'SNR':>7}"
        )
        for tag, N, K in LLAMA_7B + LLAMA_70B:
            us, tf, tb, snr = bench_one(M, N, K)
            tw_us, tw_tf = bench_tensorwise(M, N, K)
            ratio = tf / tw_tf * 100 if tw_tf else 0
            flag = "" if (snr is None or snr > 20) else "  LOW-SNR"
            print(
                f"{tag:<16}{N:>7}{K:>7}"
                f"{us:>9.1f}{tf:>8.0f}{tw_us:>9.1f}{tw_tf:>8.0f}{ratio:>6.0f}%{snr:>7.1f}{flag}"
            )


if __name__ == "__main__":
    main()
