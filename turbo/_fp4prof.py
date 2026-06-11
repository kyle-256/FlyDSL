import os, sys, torch
_R = "/workspace/code/FlyDSL"
for p in (_R, _R + "/flydsl/src"):
    sys.path.insert(0, p)
import flydsl.compiler as flyc
from turbo.mxfp4_gemm_8wave import compile_mxfp4_gemm_8w, preshuffle_mxfp4_scales

SB = 32
M, N, K = (int(x) for x in (sys.argv[1:4] or [4096, 4096, 32768]))
NITER = int(os.environ.get("FP4_PROF_ITER", "10"))
d = "cuda"
a = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device=d)
b = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=d)
asc = torch.randint(125, 130, (M, K // SB), dtype=torch.uint8, device=d)
bsc = torch.randint(125, 130, (N, K // SB), dtype=torch.uint8, device=d)
c = torch.zeros((M, N), dtype=torch.bfloat16, device=d)
asp, bsp = preshuffle_mxfp4_scales(asc, bsc, K, BLOCK_M=256, BLOCK_N=256, mode=os.environ.get("FP4_MODE","pipe"))
ai = a.view(torch.int8).view(-1); bi = b.view(torch.int8).view(-1)
st = torch.cuda.current_stream()
_pad = int(os.environ.get("FP4_PAD", "0"))
fn = compile_mxfp4_gemm_8w(K=K, BLOCK_M=256, BLOCK_N=256, mode=os.environ.get("FP4_MODE", "pipe"), block_k=int(os.environ.get("FP4_BLOCK_K", "128")), padded=_pad > 0, pad_bytes=max(_pad, 16), group_n=int(os.environ.get("GROUP_N", "0")), const_scale=os.environ.get("FP4_CONST","0")=="1", swizzle=os.environ.get("FP4_SWIZ","0")=="1", nog2s=os.environ.get("FP4_NOG2S","0")=="1")
ar = (ai, bi, c.view(-1), asp.view(-1), bsp.view(-1), M, N, st)
cc = flyc.compile(fn, *ar)
for _ in range(3):
    cc(*ar)
torch.cuda.synchronize()
_e0 = torch.cuda.Event(enable_timing=True); _e1 = torch.cuda.Event(enable_timing=True)
_best = 1e9
for _ in range(int(os.environ.get("FP4_REPS", "5"))):
    _e0.record()
    for _ in range(NITER):
        cc(*ar)
    _e1.record(); torch.cuda.synchronize()
    _best = min(_best, _e0.elapsed_time(_e1) / NITER)
_tf = 2 * M * N * K / (_best * 1e-3) / 1e12
print(f"PAD={_pad} CONST={os.environ.get('FP4_CONST','0')} M{M} N{N} K{K}  {_best:.4f} ms  {_tf:.1f} TF")
