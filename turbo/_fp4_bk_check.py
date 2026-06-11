import os, sys, torch
_R = "/workspace/code/FlyDSL"
for p in (_R, _R + "/flydsl/src"):
    sys.path.insert(0, p)
import flydsl.compiler as flyc
from turbo.mxfp4_gemm_8wave import compile_mxfp4_gemm_8w, preshuffle_mxfp4_scales

SB = 32
M, N, K = (int(x) for x in (sys.argv[1:4] or [8192, 8192, 28672]))
d = "cuda"
torch.manual_seed(0)
a = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device=d)
b = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=d)
asc = torch.randint(125, 130, (M, K // SB), dtype=torch.uint8, device=d)
bsc = torch.randint(125, 130, (N, K // SB), dtype=torch.uint8, device=d)
ai = a.view(torch.int8).view(-1); bi = b.view(torch.int8).view(-1)
st = torch.cuda.current_stream()


def run(bk):
    asp, bsp = preshuffle_mxfp4_scales(asc, bsc, K, BLOCK_M=256, BLOCK_N=256, mode="pipe")
    c = torch.zeros((M, N), dtype=torch.bfloat16, device=d)
    fn = compile_mxfp4_gemm_8w(K=K, BLOCK_M=256, BLOCK_N=256, mode="pipe", block_k=bk)
    ar = (ai, bi, c.view(-1), asp.view(-1), bsp.view(-1), M, N, st)
    cc = flyc.compile(fn, *ar)
    cc(*ar); torch.cuda.synchronize()
    return c.float().clone()


c128 = run(128)
c256 = run(256)
err = (c128 - c256)
sig = c128.norm().item()
snr = 20 * torch.log10(torch.tensor(sig / (err.norm().item() + 1e-30))).item()
print(f"M{M} N{N} K{K}  BK128 vs BK256: SNR={snr:.1f} dB  max|d|={err.abs().max().item():.4g}  "
      f"|c128|={c128.abs().mean().item():.4g}")
print("RESULT:", "OK (numerically equal)" if snr > 40 else "SUSPECT")
