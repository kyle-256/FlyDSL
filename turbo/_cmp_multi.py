"""Multi-config head-to-head: my FlyDSL variants vs competitor a4w4, SAME do_bench."""
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
asp = preshuffle_scale(asc, K, 4); bsp = preshuffle_scale_b_comb(bsc, K)
ai = a.view(torch.int8).view(-1); bi = b.view(torch.int8).view(-1); st = torch.cuda.current_stream()


def tf(ms):
    return 2 * M * N * K * 1e-12 / (ms * 1e-3)


# reference for SNR
def mxfp4_to_f32_ref(x):
    x = x.repeat_interleave(2, dim=1).clone()
    x[:, ::2] = x[:, ::2] & 0xF
    x[:, 1::2] = x[:, 1::2] >> 4
    lut = torch.tensor([0,.5,1,1.5,2,3,4,6,-0,-.5,-1,-1.5,-2,-3,-4,-6], dtype=torch.float32, device=x.device)
    return lut[x.long()]

def ref():
    af = mxfp4_to_f32_ref(a); bf = mxfp4_to_f32_ref(b)
    asf = (2.0 ** (asc.repeat_interleave(SB, 1).float() - 127))
    bsf = (2.0 ** (bsc.repeat_interleave(SB, 1).float() - 127))
    return (af * asf) @ (bf * bsf).T

REF = ref()
def snr(out):
    e = (out.float() - REF); s = (REF**2).mean(); n = (e**2).mean()
    return 10 * torch.log10(s / n).item()

CONFIGS = [
    dict(mode="pipe"),
    dict(mode="pipeh"),
    dict(mode="pipeh", block_k=256),
]
for cfg in CONFIGS:
    try:
        c = torch.zeros((M, N), dtype=torch.bfloat16, device=d)
        fn = compile_mxfp4_gemm_8w(K=K, BLOCK_M=256, BLOCK_N=256, **cfg)
        ar = (ai, bi, c.view(-1), asp.view(-1), bsp.view(-1), M, N, st)
        cc = flyc.compile(fn, *ar); cc(*ar); torch.cuda.synchronize()
        sn = snr(c)
        ms = min(triton.testing.do_bench(lambda: cc(*ar)) for _ in range(3))
        print(f"MINE {str(cfg):36s} : {tf(ms):.0f} TF ({ms*1000:.1f}us)  SNR={sn:.1f}dB")
    except Exception as e:
        print(f"MINE {str(cfg):36s} : FAIL {repr(e)[:90]}")

from matmul_kernel import matmul
matmul(a, b, asc, bsc); torch.cuda.synchronize()
ms2 = min(triton.testing.do_bench(lambda: matmul(a, b, asc, bsc)) for _ in range(3))
print(f"COMP a4w4 (BM256 BN256 BK256 w4)            : {tf(ms2):.0f} TF ({ms2*1000:.1f}us)")
