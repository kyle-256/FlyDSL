"""round-85 dynamic PMC: production-routed (recommend_config) 70B q/o 8192^3 bulk
square — the bulk geomean-driver regime. Pure kernel x10 for rocprof-compute to
sample dynamic MFMA Util% / Dependency-Wait / Wavefront Occupancy / VMEM Util%.
r34/r65 only static-counted ISA (s_barrier 513 / s_mfma 2048 / s_setprio 510);
this is the never-measured dynamic execution picture (how far from SoL)."""
import sys, torch
_R = "/workspace/code/FlyDSL"
for p in (_R, _R + "/flydsl/src"):
    sys.path.insert(0, p)
import flydsl.compiler as flyc
from turbo.mxfp4_gemm_8wave import compile_mxfp4_gemm_8w, recommend_config
from turbo.mxfp8_gemm_8wave import preshuffle_scale, preshuffle_scale_b_comb
SB = 32


def ascale(asc, M, K, BM):
    nta = BM // 64
    q = 16 * nta
    pad = ((M + q - 1) // q) * q
    if pad != M:
        ap = torch.zeros((pad, K // SB), dtype=torch.uint8, device=asc.device)
        ap[:M] = asc
        asc = ap
    return preshuffle_scale(asc, K, nta).view(-1)


M, N, K = 8192, 8192, 8192  # 70B q/o, bulk square, geomean driver, architecture-bound
d = "cuda"
a = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device=d)
b = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=d)
asc = torch.randint(125, 130, (M, K // SB), dtype=torch.uint8, device=d)
bsc = torch.randint(125, 130, (N, K // SB), dtype=torch.uint8, device=d)
BM, BN, gm, gn, nx = recommend_config(M, N, K)
print("cfg", BM, BN, gm, gn, nx, flush=True)
asp = ascale(asc, M, K, BM)
bsp = (preshuffle_scale_b_comb(bsc, K) if BN >= 256 else preshuffle_scale(bsc, K, BN // 128)).view(-1)
c = torch.zeros((M, N), dtype=torch.bfloat16, device=d)
st = torch.cuda.current_stream()
fn = compile_mxfp4_gemm_8w(K=K, BLOCK_M=BM, BLOCK_N=BN, mode="pipe", group_m=gm, group_n=gn, num_xcds=nx)
ar = (a.view(torch.int8).view(-1), b.view(torch.int8).view(-1), c.view(-1), asp, bsp, M, N, st)
cc = flyc.compile(fn, *ar)
cc(*ar)
torch.cuda.synchronize()
for _ in range(10):
    cc(*ar)
torch.cuda.synchronize()
print("done", flush=True)
