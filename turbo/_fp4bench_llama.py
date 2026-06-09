"""Dense mxfp4 GEMM bench at Llama 7B/70B shapes, M in {4096, 8192}.
Uses the turbo bench method (cuda-event hot best-of-3), production mode=pipe."""
import sys, os, torch
_R = "/workspace/code/FlyDSL"
for p in (_R, _R + "/flydsl/src"):
    sys.path.insert(0, p)
import flydsl.compiler as flyc
from turbo.mxfp4_gemm_8wave import compile_mxfp4_gemm_8w
from turbo.mxfp8_gemm_8wave import preshuffle_scale, preshuffle_scale_b_comb

SB = 32
MODE = os.environ.get("FP4_MODE", "pipe")


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


def run(tag, M, N, K):
    d = "cuda"
    a = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device=d)
    b = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=d)
    asc = torch.randint(125, 130, (M, K // SB), dtype=torch.uint8, device=d)
    bsc = torch.randint(125, 130, (N, K // SB), dtype=torch.uint8, device=d)
    c = torch.zeros((M, N), dtype=torch.bfloat16, device=d)
    asp = preshuffle_scale(asc, K, 4); bsp = preshuffle_scale_b_comb(bsc, K)
    ai = a.view(torch.int8).view(-1); bi = b.view(torch.int8).view(-1)
    st = torch.cuda.current_stream()
    fn = compile_mxfp4_gemm_8w(K=K, BLOCK_M=256, BLOCK_N=256, mode=MODE,
                               block_k=int(os.environ.get("FP4_BLOCK_K", "128")),
                               group_m=int(os.environ.get("GROUP_M", "4")),
                               num_xcds=int(os.environ.get("NUM_XCD", "8")))
    ar = (ai, bi, c.view(-1), asp.view(-1), bsp.view(-1), M, N, st)
    cc = flyc.compile(fn, *ar); cc(*ar); torch.cuda.synchronize()
    us = bench(lambda: cc(*ar))
    tf = 2 * M * N * K / (us / 1e6) / 1e12
    print(f"  {tag:12s} M={M:5d} N={N:5d} K={K:5d}: {tf:5.0f} TF ({us:7.1f}us)")


SHAPES = [
    ("7B q/o", 4096, 4096),
    ("7B gate/up", 11008, 4096),
    ("7B down", 4096, 11008),
    ("70B q/o", 8192, 8192),
    ("70B kv", 1024, 8192),
    ("70B gate/up", 28672, 8192),
    ("70B down", 8192, 28672),
]

print(f"=== dense mxfp4 (mode={MODE}, hot best-of-3) ===")
for M in (4096, 8192):
    print(f"--- M={M} ---")
    for tag, N, K in SHAPES:
        run(tag, M, N, K)
