"""r_k6 disasm: dump pipe-BN128 kv ISA to inspect main-loop vmem (buffer_load_lds)
+ s_waitcnt vmcnt counts before tightening the conservative wait_barrier(0)."""
import os, sys, torch
os.environ["FLYDSL_DUMP_IR"] = "1"
os.environ["FLYDSL_DUMP_DIR"] = "/tmp/dscan_bn128"
os.environ["FLYDSL_RUNTIME_ENABLE_CACHE"] = "0"
_R = "/workspace/code/FlyDSL"
for p in (_R, _R + "/flydsl/src"):
    sys.path.insert(0, p)
import flydsl.compiler as flyc
from turbo.mxfp4_gemm_8wave import compile_mxfp4_gemm_8w
from turbo.mxfp8_gemm_8wave import preshuffle_scale
SB = 32
M, N, K = 4096, 1024, 8192
d = "cuda"
a = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device=d)
b = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=d)
asc = torch.randint(125, 130, (M, K // SB), dtype=torch.uint8, device=d)
bsc = torch.randint(125, 130, (N, K // SB), dtype=torch.uint8, device=d)
asp = preshuffle_scale(asc, K, 256 // 64); bsp = preshuffle_scale(bsc, K, 1)
c = torch.zeros((M, N), dtype=torch.bfloat16, device=d); st = torch.cuda.current_stream()
fn = compile_mxfp4_gemm_8w(K=K, BLOCK_M=256, BLOCK_N=128, mode="pipe")
args = (a.view(torch.int8).view(-1), b.view(torch.int8).view(-1), c.view(-1), asp.view(-1), bsp.view(-1), M, N, st)
cc = flyc.compile(fn, *args); cc(*args); torch.cuda.synchronize()
print("dumped to /tmp/dscan_bn128")
