"""4-wave pipe head-to-head vs competitor, SAME do_bench, with SNR."""
import sys, torch
_R = "/workspace/code/FlyDSL"
for p in (_R, _R + "/flydsl/src"):
    sys.path.insert(0, p)
sys.path.insert(0, "/workspace/code/gfx950-gluon-tutorials/kernels/gemm/a4w4")
import triton
import flydsl.compiler as flyc
from turbo.mxfp4_gemm_4wave import compile_mxfp4_gemm_4w
from turbo.mxfp8_gemm_8wave import preshuffle_scale

M, N, K = 4096, 4096, 32768
SB = 32
d = "cuda"
a = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device=d)
b = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=d)
asc = torch.randint(125, 130, (M, K // SB), dtype=torch.uint8, device=d)
bsc = torch.randint(125, 130, (N, K // SB), dtype=torch.uint8, device=d)
NT = 256 // 4 // 16  # 4
asp = preshuffle_scale(asc, K, NT); bsp = preshuffle_scale(bsc, K, NT)
ai = a.view(torch.int8).view(-1); bi = b.view(torch.int8).view(-1); st = torch.cuda.current_stream()


def tf(ms):
    return 2 * M * N * K * 1e-12 / (ms * 1e-3)


def mxfp4_to_f32_ref(x):
    x = x.repeat_interleave(2, dim=1).clone()
    x[:, ::2] = x[:, ::2] & 0xF
    x[:, 1::2] = x[:, 1::2] >> 4
    lut = torch.tensor([0,.5,1,1.5,2,3,4,6,-0,-.5,-1,-1.5,-2,-3,-4,-6], dtype=torch.float32, device=x.device)
    return lut[x.long()]

af = mxfp4_to_f32_ref(a); bf = mxfp4_to_f32_ref(b)
asf = (2.0 ** (asc.repeat_interleave(SB, 1).float() - 127))
bsf = (2.0 ** (bsc.repeat_interleave(SB, 1).float() - 127))
REF = (af * asf) @ (bf * bsf).T
def snr(out):
    e = (out.float() - REF); s = (REF**2).mean(); n = (e**2).mean()
    return 10 * torch.log10(s / n).item()

CONFIGS = [
    dict(interleave=True),                          # current best 4-wave (3001)
    dict(mode="pipe", block_k=128),
    dict(mode="pipe", block_k=256),
]
for cfg in CONFIGS:
    try:
        c = torch.zeros((M, N), dtype=torch.bfloat16, device=d)
        fn = compile_mxfp4_gemm_4w(K=K, BLOCK_M=256, BLOCK_N=256, **cfg)
        ar = (ai, bi, c.view(-1), asp.view(-1), bsp.view(-1), M, N, st)
        cc = flyc.compile(fn, *ar); cc(*ar); torch.cuda.synchronize()
        sn = snr(c)
        ms = min(triton.testing.do_bench(lambda: cc(*ar)) for _ in range(3))
        print(f"4W {str(cfg):34s} : {tf(ms):.0f} TF ({ms*1000:.1f}us)  SNR={sn:.1f}dB")
    except Exception as e:
        print(f"4W {str(cfg):34s} : FAIL {repr(e)[:100]}")

from bench import generate_mxfp4_inputs
from matmul_kernel import matmul
ca, cb, cas, cbs = generate_mxfp4_inputs(M, N, K)
matmul(ca, cb, cas, cbs); torch.cuda.synchronize()
ms2 = min(triton.testing.do_bench(lambda: matmul(ca, cb, cas, cbs)) for _ in range(3))
print(f"COMP a4w4 (own inputs)             : {tf(ms2):.0f} TF ({ms2*1000:.1f}us)")
