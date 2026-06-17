"""500-sample benchmark to find true best dispatch time."""
import sys, torch, statistics, os
_R = "/workspace/code/FlyDSL"
for p in (_R, _R + "/flydsl/src"):
    sys.path.insert(0, p)
import flydsl.compiler as flyc
from turbo.mxfp4_gemm_4wave import compile_mxfp4_gemm_4w, preshuffle_mxfp4_scales_4w
M = int(os.environ.get("BM", "8192"))
N = int(os.environ.get("BN", "8192"))
K = int(os.environ.get("BK", "28672"))
SB = 32
d = "cuda"; torch.manual_seed(0)
a = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device=d)
b = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=d)
asc = torch.randint(125, 130, (M, K // SB), dtype=torch.uint8, device=d)
bsc = torch.randint(125, 130, (N, K // SB), dtype=torch.uint8, device=d)
ai = a.view(torch.int8).view(-1); bi = b.view(torch.int8).view(-1)
st = torch.cuda.current_stream()
asp, bsp = preshuffle_mxfp4_scales_4w(asc, bsc, K)
c = torch.zeros((M, N), dtype=torch.bfloat16, device=d)
fn = compile_mxfp4_gemm_4w(K=K)
ar = (ai, bi, c.view(-1), asp.view(-1), bsp.view(-1), M, N, st)
cc = flyc.compile(fn, *ar)
for _ in range(50):
    cc(*ar)
torch.cuda.synchronize()
N_EV = 500
ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(N_EV)]
for e0, e1 in ev:
    e0.record(); cc(*ar); e1.record()
torch.cuda.synchronize()
ts = sorted(e0.elapsed_time(e1) for e0, e1 in ev)
tf = lambda ms: 2 * M * N * K / (ms * 1e-3) / 1e12
print("500-sample M%dN%dK%d: min=%.1f p5=%.1f p10=%.1f med=%.1f TF" % (
    M, N, K, tf(ts[0]), tf(ts[24]), tf(ts[49]), tf(statistics.median(ts))))
