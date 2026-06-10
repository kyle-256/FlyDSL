"""Round-3 asm-AGPR feasibility spike: compile mxfp4 pipe with asm_mfma=True
(accumulators -> AGPR via =a inplace), check (1) VGPR/AGPR/spill vs baseline
(228/0/0), (2) correctness SNR vs the asm_mfma=False baseline output on SHARED
random fp4 input. Decides whether the deep-prefetch prerequisite (free VGPR by
moving accumulators to AGPR) is achievable."""
import os
os.environ["FLYDSL_DUMP_IR"] = "1"
os.environ["FLYDSL_DUMP_DIR"] = "/tmp/fp4agpr"
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
torch.manual_seed(0)
a = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device=d)
b = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=d)
asc = torch.randint(125, 130, (M, K // SB), dtype=torch.uint8, device=d)
bsc = torch.randint(125, 130, (N, K // SB), dtype=torch.uint8, device=d)
asp = preshuffle_scale(asc, K, 4); bsp = preshuffle_scale_b_comb(bsc, K)
ai = a.view(torch.int8).view(-1); bi = b.view(torch.int8).view(-1); st = torch.cuda.current_stream()

def run(asm):
    c = torch.zeros((M, N), dtype=torch.bfloat16, device=d)
    fn = compile_mxfp4_gemm_8w(K=K, BLOCK_M=256, BLOCK_N=256, mode="pipe", group_m=4, group_n=0, asm_mfma=asm)
    ar = (ai, bi, c.view(-1), asp.view(-1), bsp.view(-1), M, N, st)
    cc = flyc.compile(fn, *ar); cc(*ar); torch.cuda.synchronize()
    return c.clone()

def snr(o, r):
    o = o.float(); r = r.float()
    p = (r * r).mean(); n = ((o - r) ** 2).mean()
    return float("inf") if n == 0 else 10 * torch.log10(p / n).item()

c_ref = run(False)            # baseline (correct, AGPR=0)
c_asm = run(True)             # asm-inplace AGPR (dumped last -> /tmp/fp4agpr ISA = asm variant)
print(f"asm_mfma SNR vs baseline = {snr(c_asm, c_ref):.2f} dB")
print(f"c_ref[:1,:4]={c_ref[:1,:4].tolist()}")
print(f"c_asm[:1,:4]={c_asm[:1,:4].tolist()}")
print("DONE")
