"""Definitive same-process duel: my 8-wave pipe vs competitor a4w4.
Alternates launches under one process so rocprofv3 --kernel-trace captures both
at identical clock/thermal. Also verifies SNR of my kernel against fp32 ref."""
import sys, torch
_R = "/workspace/code/FlyDSL"
for p in (_R, _R + "/flydsl/src"):
    sys.path.insert(0, p)
sys.path.insert(0, "/workspace/code/gfx950-gluon-tutorials/kernels/gemm/a4w4")
import flydsl.compiler as flyc
from turbo.mxfp4_gemm_8wave import compile_mxfp4_gemm_8w
from turbo.mxfp8_gemm_8wave import preshuffle_scale, preshuffle_scale_b_comb
from matmul_kernel import matmul
from bench import generate_mxfp4_inputs

M, N, K = 4096, 4096, 32768
SB = 32
d = "cuda"
torch.manual_seed(0)
a = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device=d)
b = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=d)
asc = torch.randint(125, 130, (M, K // SB), dtype=torch.uint8, device=d)
bsc = torch.randint(125, 130, (N, K // SB), dtype=torch.uint8, device=d)

# my kernel
c = torch.zeros((M, N), dtype=torch.bfloat16, device=d)
asp = preshuffle_scale(asc, K, 4); bsp = preshuffle_scale_b_comb(bsc, K)
ai = a.view(torch.int8).view(-1); bi = b.view(torch.int8).view(-1); st = torch.cuda.current_stream()
fn = compile_mxfp4_gemm_8w(K=K, BLOCK_M=256, BLOCK_N=256, mode="pipe")
ar = (ai, bi, c.view(-1), asp.view(-1), bsp.view(-1), M, N, st)
cc = flyc.compile(fn, *ar); cc(*ar); torch.cuda.synchronize()

# SNR of my kernel
def mxfp4_to_f32_ref(x):
    x = x.repeat_interleave(2, dim=1).clone()
    x[:, ::2] = x[:, ::2] & 0xF
    x[:, 1::2] = x[:, 1::2] >> 4
    lut = torch.tensor([0,.5,1,1.5,2,3,4,6,-0,-.5,-1,-1.5,-2,-3,-4,-6], dtype=torch.float32, device=x.device)
    return lut[x.long()]
af = mxfp4_to_f32_ref(a); bf = mxfp4_to_f32_ref(b)
asf = (2.0 ** (asc.repeat_interleave(SB, 1).float() - 127)); bsf = (2.0 ** (bsc.repeat_interleave(SB, 1).float() - 127))
REF = (af * asf) @ (bf * bsf).T
e = (c.float() - REF); snr = 10 * torch.log10((REF**2).mean() / (e**2).mean()).item()
print(f"MINE SNR = {snr:.1f} dB  (correctness)")

# competitor with ITS OWN correct input layout (column-major scales etc.)
ca, cb, cas, cbs = generate_mxfp4_inputs(M, N, K)
matmul(ca, cb, cas, cbs); torch.cuda.synchronize()

# warmup
for _ in range(10):
    cc(*ar); matmul(ca, cb, cas, cbs)
torch.cuda.synchronize()

# alternate launches (rocprofv3 captures both, same clock/thermal)
for _ in range(150):
    cc(*ar)
    matmul(ca, cb, cas, cbs)
torch.cuda.synchronize()
print("duel done")
