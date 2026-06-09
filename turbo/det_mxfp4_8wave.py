#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Determinism (race) red-line check for turbo/mxfp4_gemm_8wave.py.

det == 0 means bit-exact across runs == race-free. FRESH random packed-fp4
a/b + E8M0 scales each pass; multiple shapes (big-K / big-N). Pick impl via
FP4_MODE env (pipe / staged / direct).

Run:  python turbo/det_mxfp4_8wave.py [runs] [passes]
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
from turbo.mxfp4_gemm_8wave import compile_mxfp4_gemm_8w  # noqa: E402
from turbo.mxfp8_gemm_8wave import preshuffle_scale, preshuffle_scale_b_comb  # noqa: E402

SCALE_BLOCK = 32
MODE = os.environ.get("FP4_MODE", "pipe")

SHAPES = [
    ("square_sm   ", 256, 256, 512),
    ("square_big  ", 4096, 4096, 4096),
    ("big_K       ", 4096, 4096, 11008),
    ("big_N       ", 4096, 28672, 8192),
    ("peak        ", 4096, 4096, 32768),
]


def _mk(M, N, K, dev="cuda"):
    a = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device=dev)
    b = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=dev)
    asc = torch.randint(124, 131, (M, K // SCALE_BLOCK), dtype=torch.uint8, device=dev)
    bsc = torch.randint(124, 131, (N, K // SCALE_BLOCK), dtype=torch.uint8, device=dev)
    return a, b, asc, bsc


def det_one(M, N, K, runs, passes, BLOCK_M=256, BLOCK_N=256):
    dev = "cuda"
    launch_fn = compile_mxfp4_gemm_8w(
        K=K, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, mode=MODE, block_k=int(os.environ.get("FP4_BLOCK_K", "128"))
    )
    worst = 0.0
    first_bad = (-1, -1)
    for pi in range(passes):
        a, b, asc, bsc = _mk(M, N, K, dev)
        a_sp = preshuffle_scale(asc, K, BLOCK_M // 64)
        b_sp = preshuffle_scale_b_comb(bsc, K)
        c = torch.zeros((M, N), dtype=torch.bfloat16, device=dev)

        def args(cc):
            return (
                a.view(torch.int8).view(-1),
                b.view(torch.int8).view(-1),
                cc.view(-1),
                a_sp.view(-1),
                b_sp.view(-1),
                M,
                N,
                torch.cuda.current_stream(),
            )

        compiled = flyc.compile(launch_fn, *args(c))
        compiled(*args(c))
        torch.cuda.synchronize()
        ref = c.clone()
        for ri in range(runs):
            c.zero_()
            compiled(*args(c))
            d = (c.float() - ref.float()).abs().max().item()
            if d > worst:
                worst = d
            if d > 0 and first_bad[0] < 0:
                first_bad = (pi, ri)
                break
        if first_bad[0] >= 0:
            break
    return worst, first_bad


def main():
    arch = str(get_rocm_arch())
    assert "gfx95" in arch, f"needs gfx950, got {arch}"
    argv = [int(x) for x in sys.argv[1:]]
    runs = argv[0] if len(argv) >= 1 else 2000
    passes = argv[1] if len(argv) >= 2 else 3
    print(f"det check mode={MODE}: runs={runs}/pass, passes={passes}, fresh data each pass\n")
    print(f"{'shape':<14}{'M':>6}{'N':>7}{'K':>7}{'max_diff':>12}{'first_bad':>14}  result")
    all_ok = True
    for tag, M, N, K in SHAPES:
        worst, fb = det_one(M, N, K, runs, passes)
        ok = worst == 0.0
        all_ok = all_ok and ok
        fbs = "-" if fb[0] < 0 else f"p{fb[0]}r{fb[1]}"
        print(f"{tag:<14}{M:>6}{N:>7}{K:>7}{worst:>12.2e}{fbs:>14}  {'DET0 ✓' if ok else 'RACE ✗'}")
    print("\nOVERALL:", "DET0 (race-free)" if all_ok else "RACE DETECTED")


if __name__ == "__main__":
    main()
