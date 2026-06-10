"""Round-34 Rule7-L3 ISA inspection: dump production (r6_4) ISA for a bulk shape
(70B q/o 8192x8192x8192, BM256/BN256/gm4/gn0 = the bulk-regime production config)
and count v_mfma / s_barrier / ds_read / buffer_load / s_waitcnt + VGPR/AGPR/spill,
to quantify the source-level ceiling (gemm-opt SKILL Section 9). No kernel change."""
import os, sys, glob, re
os.environ["FLYDSL_DUMP_IR"] = "1"
os.environ["FLYDSL_DUMP_DIR"] = "/tmp/fp4isa"
os.environ["FLYDSL_RUNTIME_ENABLE_CACHE"] = "0"
os.system("rm -rf /tmp/fp4isa")
import torch
_R = "/workspace/code/FlyDSL"
for p in (_R, _R + "/flydsl/src"):
    sys.path.insert(0, p)
import flydsl.compiler as flyc
from turbo.mxfp4_gemm_8wave import compile_mxfp4_gemm_8w
from turbo.mxfp8_gemm_8wave import preshuffle_scale, preshuffle_scale_b_comb
SB = 32

M, N, K = 8192, 8192, 8192  # 70B q/o, bulk square (production BM256/BN256/gm4/gn0)
d = "cuda"
a = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device=d)
b = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=d)
asc = torch.randint(125, 130, (M, K // SB), dtype=torch.uint8, device=d)
bsc = torch.randint(125, 130, (N, K // SB), dtype=torch.uint8, device=d)
asp = preshuffle_scale(asc, K, 4).view(-1)
bsp = preshuffle_scale_b_comb(bsc, K).view(-1)
c = torch.zeros((M, N), dtype=torch.bfloat16, device=d)
st = torch.cuda.current_stream()
fn = compile_mxfp4_gemm_8w(K=K, BLOCK_M=256, BLOCK_N=256, mode="pipe", group_m=4, group_n=0)
ar = (a.view(torch.int8).view(-1), b.view(torch.int8).view(-1), c.view(-1), asp, bsp, M, N, st)
cc = flyc.compile(fn, *ar); cc(*ar); torch.cuda.synchronize()
print("compiled+ran OK", flush=True)

isa = glob.glob("/tmp/fp4isa/**/*final_isa.s", recursive=True)
print("ISA files:", isa, flush=True)
if not isa:
    print("NO ISA dumped", flush=True); sys.exit(1)
# pick the largest (main gemm kernel)
f = max(isa, key=lambda p: os.path.getsize(p))
txt = open(f).read()
lines = txt.splitlines()
def cnt(pat):
    return sum(1 for l in lines if re.search(pat, l))
print(f"--- ISA {f} ({len(lines)} lines) ---", flush=True)
print(f"v_mfma            : {cnt(r'v_mfma')}", flush=True)
print(f"s_barrier         : {cnt(r's_barrier')}", flush=True)
print(f"ds_read           : {cnt(r'ds_read')}", flush=True)
print(f"buffer_load       : {cnt(r'buffer_load')}", flush=True)
print(f"s_waitcnt(any)    : {cnt(r's_waitcnt')}", flush=True)
print(f"  vmcnt(0)        : {cnt(r'vmcnt\(0\)')}", flush=True)
print(f"  lgkmcnt(0)      : {cnt(r'lgkmcnt\(0\)')}", flush=True)
print(f"s_setprio         : {cnt(r's_setprio')}", flush=True)
print(f"scratch (spill)   : {cnt(r'scratch_')}", flush=True)
for l in lines:
    if re.search(r"\.vgpr_count|\.sgpr_count|\.agpr_count|vgpr_spill|accum_offset|next_free_vgpr", l):
        print("  META:", l.strip(), flush=True)
print("DONE", flush=True)
