"""Sweep group_n (2D band swizzle) for dense mxfp4 pipe on big-N Llama shapes.
Verifies bit-exact vs group_n=0 (band swizzle is a pure tile permutation)."""
import sys, os, torch
_R = "/workspace/code/FlyDSL"
for p in (_R, _R + "/flydsl/src"):
    sys.path.insert(0, p)
import flydsl.compiler as flyc
from turbo.mxfp4_gemm_8wave import compile_mxfp4_gemm_8w
from turbo.mxfp8_gemm_8wave import preshuffle_scale, preshuffle_scale_b_comb
SB = 32


def bench(fn, it=30, wu=8, reps=3):
    fn(); torch.cuda.synchronize()
    for _ in range(wu):
        fn()
    torch.cuda.synchronize(); best = 1e9
    for _ in range(reps):
        e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(); e0.record()
        for _ in range(it):
            fn()
        e1.record(); torch.cuda.synchronize(); best = min(best, e0.elapsed_time(e1) * 1000 / it)
    return best


def make(M, N, K):
    d = "cuda"
    a = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device=d)
    b = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=d)
    asc = torch.randint(125, 130, (M, K // SB), dtype=torch.uint8, device=d)
    bsc = torch.randint(125, 130, (N, K // SB), dtype=torch.uint8, device=d)
    asp = preshuffle_scale(asc, K, 4); bsp = preshuffle_scale_b_comb(bsc, K)
    ai = a.view(torch.int8).view(-1); bi = b.view(torch.int8).view(-1)
    return ai, bi, asp.view(-1), bsp.view(-1)


def run(tag, M, N, K, gn, inp, ref=None):
    d = "cuda"
    ai, bi, asp, bsp = inp
    c = torch.zeros((M, N), dtype=torch.bfloat16, device=d)
    st = torch.cuda.current_stream()
    fn = compile_mxfp4_gemm_8w(K=K, BLOCK_M=256, BLOCK_N=256, mode="pipe", group_m=4, group_n=gn)
    ar = (ai, bi, c.view(-1), asp, bsp, M, N, st)
    cc = flyc.compile(fn, *ar); cc(*ar); torch.cuda.synchronize()
    out = c.clone()
    us = bench(lambda: cc(*ar))
    tf = 2 * M * N * K / (us / 1e6) / 1e12
    ok = "" if ref is None else ("  EXACT" if torch.equal(out, ref) else "  !!DIFF")
    return tf, out, ok


SHAPES = [
    ("7B gate/up", 4096, 11008, 4096),
    ("70B gate/up", 4096, 28672, 8192),
    ("70B down", 4096, 8192, 28672),
    ("70B q/o", 4096, 8192, 8192),
]
for tag, M, N, K in SHAPES:
    nb = N // 256
    GNS = sorted({max(1, nb // 8), max(1, nb // 4), max(1, nb // 2)})
    inp = make(M, N, K)
    tf0, ref, _ = run(tag, M, N, K, 0, inp)
    line = f"{tag:13s} nb={nb:3d}  gn=0:{tf0:5.0f}"
    for gn in GNS:
        tf, _, ok = run(tag, M, N, K, gn, inp, ref)
        line += f"  gn={gn}:{tf:5.0f}{ok}"
    print(line)
