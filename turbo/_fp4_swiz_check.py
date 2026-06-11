import os, sys, torch
_R = "/workspace/code/FlyDSL"
for p in (_R, _R + "/flydsl/src"):
    sys.path.insert(0, p)
import flydsl.compiler as flyc
from turbo.mxfp4_gemm_8wave import compile_mxfp4_gemm_8w, preshuffle_mxfp4_scales

SB = 32
M, N, K = (int(x) for x in (sys.argv[1:4] or [4096, 4096, 16384]))
d = "cuda"
torch.manual_seed(0)
a = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device=d)
b = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=d)
asc = torch.randint(125, 130, (M, K // SB), dtype=torch.uint8, device=d)
bsc = torch.randint(125, 130, (N, K // SB), dtype=torch.uint8, device=d)
ai = a.view(torch.int8).view(-1); bi = b.view(torch.int8).view(-1)
st = torch.cuda.current_stream()


BK = int(os.environ.get("FP4_BLOCK_K", "128"))


def run(swz):
    asp, bsp = preshuffle_mxfp4_scales(asc, bsc, K, BLOCK_M=256, BLOCK_N=256, mode="pipe")
    c = torch.zeros((M, N), dtype=torch.bfloat16, device=d)
    fn = compile_mxfp4_gemm_8w(K=K, BLOCK_M=256, BLOCK_N=256, mode="pipe", block_k=BK, swizzle=swz)
    ar = (ai, bi, c.view(-1), asp.view(-1), bsp.view(-1), M, N, st)
    cc = flyc.compile(fn, *ar)
    cc(*ar); torch.cuda.synchronize()
    return c.clone()


c0 = run(False)
c1 = run(True)
diff = (c0.float() - c1.float()).abs()
print(f"M{M} N{N} K{K}  identity vs swizzle: max|diff|={diff.max().item():.6g}  "
      f"mean|diff|={diff.mean().item():.6g}  nonzero={int((diff>0).sum().item())}/{diff.numel()}")
print("RESULT:", "BIT-EXACT PASS" if diff.max().item() == 0 else "MISMATCH FAIL")
