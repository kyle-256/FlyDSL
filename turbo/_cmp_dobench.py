"""Apples-to-apples: my FlyDSL pipe vs competitor a4w4, SAME do_bench, SAME shape."""
import sys, torch
_R = "/workspace/code/FlyDSL"
for p in (_R, _R + "/flydsl/src"):
    sys.path.insert(0, p)
sys.path.insert(0, "/workspace/code/gfx950-gluon-tutorials/kernels/gemm/a4w4")
import triton
import flydsl.compiler as flyc
from turbo.mxfp4_gemm_8wave import compile_mxfp4_gemm_8w
from turbo.mxfp8_gemm_8wave import preshuffle_scale, preshuffle_scale_b_comb

M, N, K = 4096, 4096, 32768
SB = 32
d = "cuda"
a = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device=d)
b = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=d)
asc = torch.randint(125, 130, (M, K // SB), dtype=torch.uint8, device=d)
bsc = torch.randint(125, 130, (N, K // SB), dtype=torch.uint8, device=d)


def tf(ms):
    return 2 * M * N * K * 1e-12 / (ms * 1e-3)


# --- mine (pipe) ---
c = torch.zeros((M, N), dtype=torch.bfloat16, device=d)
asp = preshuffle_scale(asc, K, 4); bsp = preshuffle_scale_b_comb(bsc, K)
ai = a.view(torch.int8).view(-1); bi = b.view(torch.int8).view(-1); st = torch.cuda.current_stream()
fn = compile_mxfp4_gemm_8w(K=K, BLOCK_M=256, BLOCK_N=256, mode="pipe")
ar = (ai, bi, c.view(-1), asp.view(-1), bsp.view(-1), M, N, st)
cc = flyc.compile(fn, *ar); cc(*ar); torch.cuda.synchronize()
ms = triton.testing.do_bench(lambda: cc(*ar))
print(f"MINE  pipe  do_bench: {tf(ms):.0f} TF ({ms*1000:.1f}us)")

# --- competitor ---
try:
    from matmul_kernel import matmul
    ms2 = triton.testing.do_bench(lambda: matmul(a, b, asc, bsc))
    print(f"COMP  a4w4  do_bench: {tf(ms2):.0f} TF ({ms2*1000:.1f}us)")
except Exception as e:
    print("comp fail:", e)
