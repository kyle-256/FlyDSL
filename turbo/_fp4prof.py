import os, sys, torch
_R = "/workspace/code/FlyDSL"
for p in (_R, _R + "/flydsl/src"):
    sys.path.insert(0, p)
import flydsl.compiler as flyc
from turbo.mxfp4_gemm_8wave import compile_mxfp4_gemm_8w
from turbo.mxfp8_gemm_8wave import preshuffle_scale, preshuffle_scale_b_comb

SB = 32
M, N, K = (int(x) for x in (sys.argv[1:4] or [4096, 4096, 32768]))
NITER = int(os.environ.get("FP4_PROF_ITER", "10"))
d = "cuda"
a = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device=d)
b = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=d)
asc = torch.randint(125, 130, (M, K // SB), dtype=torch.uint8, device=d)
bsc = torch.randint(125, 130, (N, K // SB), dtype=torch.uint8, device=d)
c = torch.zeros((M, N), dtype=torch.bfloat16, device=d)
asp = preshuffle_scale(asc, K, 4); bsp = preshuffle_scale_b_comb(bsc, K)
ai = a.view(torch.int8).view(-1); bi = b.view(torch.int8).view(-1)
st = torch.cuda.current_stream()
_pad = int(os.environ.get("FP4_PAD", "0"))
fn = compile_mxfp4_gemm_8w(K=K, BLOCK_M=256, BLOCK_N=256, mode=os.environ.get("FP4_MODE", "pipe"), block_k=int(os.environ.get("FP4_BLOCK_K", "128")), padded=_pad > 0, pad_bytes=max(_pad, 16), group_n=int(os.environ.get("GROUP_N", "0")))
ar = (ai, bi, c.view(-1), asp.view(-1), bsp.view(-1), M, N, st)
cc = flyc.compile(fn, *ar)
for _ in range(3):
    cc(*ar)
torch.cuda.synchronize()
for _ in range(NITER):
    cc(*ar)
torch.cuda.synchronize()
