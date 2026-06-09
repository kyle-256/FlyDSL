"""Full Llama 7B/70B dense mxfp4 table: baseline (gn=0) vs 2D band swizzle.
For each shape tries gn in {0, nb//8, nb//4}, verifies bit-exact, reports best."""
import sys, os, torch
_R = "/workspace/code/FlyDSL"
for p in (_R, _R + "/flydsl/src"):
    sys.path.insert(0, p)
import flydsl.compiler as flyc
from turbo.mxfp4_gemm_8wave import compile_mxfp4_gemm_8w
from turbo.mxfp8_gemm_8wave import preshuffle_scale, preshuffle_scale_b_comb
SB = 32


def bench(fn, it=30, wu=8, reps=4):
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
    return a.view(torch.int8).view(-1), b.view(torch.int8).view(-1), asp.view(-1), bsp.view(-1)


def one(M, N, K, gn, inp):
    ai, bi, asp, bsp = inp
    c = torch.zeros((M, N), dtype=torch.bfloat16, device="cuda")
    st = torch.cuda.current_stream()
    fn = compile_mxfp4_gemm_8w(K=K, BLOCK_M=256, BLOCK_N=256, mode="pipe", group_m=4, group_n=gn)
    ar = (ai, bi, c.view(-1), asp, bsp, M, N, st)
    cc = flyc.compile(fn, *ar); cc(*ar); torch.cuda.synchronize()
    out = c.clone()
    us = bench(lambda: cc(*ar))
    return 2 * M * N * K / (us / 1e6) / 1e12, out


SHAPES = [
    ("7B q/o", 4096, 4096), ("7B gate/up", 11008, 4096), ("7B down", 4096, 11008),
    ("70B q/o", 8192, 8192), ("70B kv", 1024, 8192),
    ("70B gate/up", 28672, 8192), ("70B down", 8192, 28672),
]
for M in (4096, 8192):
    print(f"--- M={M} ---")
    for tag, N, K in SHAPES:
        nb = N // 256
        gns = sorted({g for g in (max(1, nb // 8), max(1, nb // 4)) if g < nb})
        inp = make(M, N, K)
        tf0, ref = one(M, N, K, 0, inp)
        best_tf, best_gn = tf0, 0
        exact = True
        for gn in gns:
            tf, out = one(M, N, K, gn, inp)
            if not torch.equal(out, ref):
                exact = False
            if tf > best_tf:
                best_tf, best_gn = tf, gn
        win = (best_tf / tf0 - 1) * 100
        flag = "" if exact else "  !!DIFF"
        print(f"  {tag:12s} N={N:5d} K={K:5d}: base {tf0:5.0f} -> best {best_tf:5.0f} "
              f"(gn={best_gn:2d}, {win:+4.1f}%){flag}")
