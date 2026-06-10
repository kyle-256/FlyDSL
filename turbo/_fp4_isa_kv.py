"""round-67 Rule7-L4 ISA: dump production (recommend_config) ISA for the kv worst shape
(70B kv 1024x8192 M=4096, routes to BM128/BN128/gm8/gn0) and count v_mfma / s_barrier /
s_setprio / ds_read / buffer_load / s_waitcnt + VGPR/AGPR/spill, to compare vs aiter kv
192x128 .co. No kernel change."""
import os, sys, glob, re
os.environ["FLYDSL_DUMP_IR"] = "1"
os.environ["FLYDSL_DUMP_DIR"] = "/tmp/fp4isakv"
os.environ["FLYDSL_RUNTIME_ENABLE_CACHE"] = "0"
os.system("rm -rf /tmp/fp4isakv")
import torch
_R = "/workspace/code/FlyDSL"
for p in (_R, _R + "/flydsl/src"):
    sys.path.insert(0, p)
import flydsl.compiler as flyc
from turbo.mxfp4_gemm_8wave import compile_mxfp4_gemm_8w, recommend_config
from turbo.mxfp8_gemm_8wave import preshuffle_scale, preshuffle_scale_b_comb
SB = 32

M, N, K = 4096, 1024, 8192  # 70B kv, worst min_ratio shape
BM, BN, gm, gn = recommend_config(M, N, K)
print(f"kv recommend_config: BM={BM} BN={BN} gm={gm} gn={gn}", flush=True)
d = "cuda"
a = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device=d)
b = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=d)
asc = torch.randint(125, 130, (M, K // SB), dtype=torch.uint8, device=d)
bsc = torch.randint(125, 130, (N, K // SB), dtype=torch.uint8, device=d)
nta = BM // 64
q = 16 * nta
pad = ((M + q - 1) // q) * q
if pad != M:
    ap = torch.zeros((pad, K // SB), dtype=torch.uint8, device=d); ap[:M] = asc; asc = ap
asp = preshuffle_scale(asc, K, nta).view(-1)
bsp = (preshuffle_scale_b_comb(bsc, K) if BN >= 256 else preshuffle_scale(bsc, K, BN // 128)).view(-1)
c = torch.zeros((M, N), dtype=torch.bfloat16, device=d)
st = torch.cuda.current_stream()
fn = compile_mxfp4_gemm_8w(K=K, BLOCK_M=BM, BLOCK_N=BN, mode="pipe", group_m=gm, group_n=gn)
ar = (a.view(torch.int8).view(-1), b.view(torch.int8).view(-1), c.view(-1), asp, bsp, M, N, st)
cc = flyc.compile(fn, *ar); cc(*ar); torch.cuda.synchronize()
print("compiled+ran OK", flush=True)

isa = glob.glob("/tmp/fp4isakv/**/*final_isa.s", recursive=True)
if not isa:
    print("NO ISA dumped", flush=True); sys.exit(1)
f = max(isa, key=lambda p: os.path.getsize(p))
lines = open(f).read().splitlines()
def cnt(pat):
    return sum(1 for l in lines if re.search(pat, l))
print(f"--- fly kv ISA {f} ({len(lines)} lines) ---", flush=True)
for nm, pat in [("v_mfma", r"v_mfma"), ("s_barrier", r"s_barrier"), ("s_setprio", r"s_setprio"),
                ("ds_read", r"ds_read"), ("buffer_load", r"buffer_load"), ("s_waitcnt", r"s_waitcnt"),
                ("  vmcnt(0)", r"vmcnt\(0\)"), ("  lgkmcnt(0)", r"lgkmcnt\(0\)"), ("s_nop", r"s_nop"),
                ("scratch", r"scratch_")]:
    print(f"{nm:14s}: {cnt(pat)}", flush=True)
for l in lines:
    if re.search(r"\.vgpr_count|\.agpr_count|\.sgpr_count|vgpr_spill|accum_offset", l):
        print("  META:", l.strip(), flush=True)
print("DONE", flush=True)
