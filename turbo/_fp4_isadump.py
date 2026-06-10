"""Dump final ISA for mxfp4 pipe kernel (square 8192^3, K=8192) to inspect the
main-loop lgkmcnt(0) drains vs ds_read/mfma adjacency (drain-removal feasibility)."""
import os
os.environ["FLYDSL_DUMP_IR"] = "1"
os.environ["FLYDSL_DUMP_DIR"] = "/tmp/fp4isa"
os.environ["FLYDSL_RUNTIME_ENABLE_CACHE"] = "0"
import sys, torch
_R = "/workspace/code/FlyDSL"
for p in (_R, _R + "/flydsl/src"):
    sys.path.insert(0, p)
import flydsl.compiler as flyc
from turbo.mxfp4_gemm_8wave import compile_mxfp4_gemm_8w
from turbo.mxfp8_gemm_8wave import preshuffle_scale, preshuffle_scale_b_comb
SB = 32
M, N, K = 8192, 8192, 8192
d = "cuda"
a = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device=d)
b = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=d)
asc = torch.randint(125, 130, (M, K // SB), dtype=torch.uint8, device=d)
bsc = torch.randint(125, 130, (N, K // SB), dtype=torch.uint8, device=d)
c = torch.zeros((M, N), dtype=torch.bfloat16, device=d)
asp = preshuffle_scale(asc, K, 4); bsp = preshuffle_scale_b_comb(bsc, K)
ai = a.view(torch.int8).view(-1); bi = b.view(torch.int8).view(-1); st = torch.cuda.current_stream()
fn = compile_mxfp4_gemm_8w(K=K, BLOCK_M=256, BLOCK_N=256, mode="pipe", group_m=4, group_n=0)
ar = (ai, bi, c.view(-1), asp.view(-1), bsp.view(-1), M, N, st)
cc = flyc.compile(fn, *ar); cc(*ar); torch.cuda.synchronize()
print("DONE compile+run")
