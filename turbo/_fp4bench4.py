import sys, os, torch
_R = "/workspace/code/FlyDSL"
for p in (_R, _R + "/flydsl/src"):
    sys.path.insert(0, p)
import flydsl.compiler as flyc
from turbo.mxfp4_gemm_4wave import compile_mxfp4_gemm_4w
from turbo.mxfp8_gemm_8wave import preshuffle_scale

SB = 32
PAD = int(os.environ.get("FP4_PAD", "0"))
PROF = int(os.environ.get("FP4_PROF_ITER", "0"))


def bench(fn, it=30, wu=8, reps=3):
    fn(); torch.cuda.synchronize()
    for _ in range(wu): fn()
    torch.cuda.synchronize(); best = 1e9
    for _ in range(reps):
        e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(); e0.record()
        for _ in range(it): fn()
        e1.record(); torch.cuda.synchronize(); best = min(best, e0.elapsed_time(e1) * 1000 / it)
    return best


def run(M, N, K):
    d = "cuda"
    a = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device=d)
    b = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=d)
    asc = torch.randint(125, 130, (M, K // SB), dtype=torch.uint8, device=d)
    bsc = torch.randint(125, 130, (N, K // SB), dtype=torch.uint8, device=d)
    c = torch.zeros((M, N), dtype=torch.bfloat16, device=d)
    asp = preshuffle_scale(asc, K, 256 // 4 // 16); bsp = preshuffle_scale(bsc, K, 256 // 4 // 16)
    ai = a.view(torch.int8).view(-1); bi = b.view(torch.int8).view(-1); st = torch.cuda.current_stream()
    fn = compile_mxfp4_gemm_4w(
        K=K, BLOCK_M=256, BLOCK_N=256, padded=PAD > 0, pad_bytes=max(PAD, 16),
        asm_mfma=int(os.environ.get("FP4_ASM", "0")) > 0,
        interleave=int(os.environ.get("FP4_IL", "0")) > 0,
        asm_se=int(os.environ.get("FP4_SE", "0")) > 0,
    )
    ar = (ai, bi, c.view(-1), asp.view(-1), bsp.view(-1), M, N, st)
    cc = flyc.compile(fn, *ar); cc(*ar); torch.cuda.synchronize()
    if PROF:
        for _ in range(3): cc(*ar)
        torch.cuda.synchronize()
        for _ in range(PROF): cc(*ar)
        torch.cuda.synchronize(); return
    us = bench(lambda: cc(*ar)); tf = 2 * M * N * K / (us / 1e6) / 1e12
    print(f"mxfp4-4w pad={PAD} {M}x{N}x{K}: {tf:.0f} TF ({us:.1f}us)  [竞品 5253]")


if len(sys.argv) >= 4:
    run(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))
else:
    run(4096, 4096, 32768)
    run(4096, 4096, 4096)
