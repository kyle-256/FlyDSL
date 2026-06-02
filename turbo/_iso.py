#!/usr/bin/env python3
"""Isolate v_mfma_scale cost vs scale-load cost.
tw       : tensorwise (plain v_mfma, no scale work)
mx       : full mxfp8 (v_mfma_scale + scale loads)
mx_const : v_mfma_scale with 0x7F7F7F7F immediate, NO scale loads
"""
import os
import sys

import torch

_R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for p in (_R, os.path.join(_R, "flydsl", "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import flydsl.compiler as flyc  # noqa: E402
from kernels.fp8_gemm_8wave import compile_fp8_gemm_8w  # noqa: E402
from turbo.mxfp8_gemm_8wave import compile_mxfp8_gemm_8w, preshuffle_scale, preshuffle_scale_b_comb  # noqa: E402


def bench(fn, it=50, warmup=10, reps=4):
    fn(); torch.cuda.synchronize()
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(reps):
        e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(); e0.record()
        for _ in range(it): fn()
        e1.record(); torch.cuda.synchronize()
        best = min(best, e0.elapsed_time(e1) * 1000.0 / it)
    return best


def run(M, N, K):
    dev = "cuda"
    a = torch.randint(0, 126, (M, K), dtype=torch.uint8, device=dev)
    b = torch.randint(0, 126, (N, K), dtype=torch.uint8, device=dev)
    c = torch.zeros((M, N), dtype=torch.bfloat16, device=dev)
    asc = torch.randint(125, 130, (M, K // 32), dtype=torch.uint8, device=dev)
    bsc = torch.randint(125, 130, (N, K // 32), dtype=torch.uint8, device=dev)
    a_sp = preshuffle_scale(asc, K, 4); b_sp = preshuffle_scale_b_comb(bsc, K)
    sa1 = torch.ones(M, dtype=torch.float32, device=dev); sb1 = torch.ones(N, dtype=torch.float32, device=dev)
    ai = a.view(torch.int8).view(-1); bi = b.view(torch.int8).view(-1); ci = c.view(-1)
    st = torch.cuda.current_stream()

    def tf(us):
        return 2 * M * N * K / (us / 1e6) / 1e12

    res = {}
    # tw
    fn = compile_fp8_gemm_8w(K=K, BLOCK_M=256, BLOCK_N=256)
    args = (ai, bi, ci, sa1.view(-1), sb1.view(-1), M, N, st)
    comp = flyc.compile(fn, *args); res["tw"] = bench(lambda: comp(*args))
    # mx normal
    fn = compile_mxfp8_gemm_8w(K=K, BLOCK_M=256, BLOCK_N=256)
    args = (ai, bi, ci, a_sp.view(-1), b_sp.view(-1), M, N, st)
    comp = flyc.compile(fn, *args); res["mx"] = bench(lambda: comp(*args))
    # mx const (no loads, v_mfma_scale immediate)
    fn = compile_mxfp8_gemm_8w(K=K, BLOCK_M=256, BLOCK_N=256, const_scale=True)
    args = (ai, bi, ci, a_sp.view(-1), b_sp.view(-1), M, N, st)
    comp = flyc.compile(fn, *args); res["mx_const"] = bench(lambda: comp(*args))

    print(f"\n{M}x{N}x{K}")
    base = res["tw"]
    for k in ("tw", "mx_const", "mx"):
        print(f"  {k:<9} {res[k]:7.1f}us  {tf(res[k]):7.0f} TF  {base/res[k]*100:5.1f}% of tw")


if __name__ == "__main__":
    for shp in [(4096, 28672, 8192), (8192, 8192, 28672), (4096, 4096, 4096)]:
        run(*shp)
