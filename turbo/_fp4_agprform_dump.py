"""round-86 r_b1': ISA-dump V/A/spill for intrinsic scale-MFMA with
amdgpu-mfma-vgpr-form=false function attr (FP4_AGPR_FORM env). Compares acc
register placement vs baseline. Bulk q/o 8192^3, mode=pipe (production)."""
import os
FORM = os.environ.get("FP4_AGPR_FORM", "1")
os.environ["FP4_AGPR_FORM"] = FORM
tag = "on" if FORM == "1" else "off"
os.environ["FLYDSL_DUMP_IR"] = "1"
os.environ["FLYDSL_DUMP_DIR"] = f"/tmp/dscan_{tag}"
os.environ["FLYDSL_RUNTIME_ENABLE_CACHE"] = "0"
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


M, N, K = 8192, 8192, 8192
d = "cuda"
a = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device=d)
b = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=d)
asc = torch.randint(125, 130, (M, K // SB), dtype=torch.uint8, device=d)
bsc = torch.randint(125, 130, (N, K // SB), dtype=torch.uint8, device=d)
BM, BN, gm, gn, nx = recommend_config(M, N, K)
print("cfg", BM, BN, gm, gn, nx, "FORM", FORM, flush=True)
asp = ascale(asc, M, K, BM)
bsp = (preshuffle_scale_b_comb(bsc, K) if BN >= 256 else preshuffle_scale(bsc, K, BN // 128)).view(-1)
c = torch.zeros((M, N), dtype=torch.bfloat16, device=d)
st = torch.cuda.current_stream()
fn = compile_mxfp4_gemm_8w(K=K, BLOCK_M=BM, BLOCK_N=BN, mode="pipe", group_m=gm, group_n=gn, num_xcds=nx)
ar = (a.view(torch.int8).view(-1), b.view(torch.int8).view(-1), c.view(-1), asp, bsp, M, N, st)
cc = flyc.compile(fn, *ar)
cc(*ar)
torch.cuda.synchronize()
print("done", flush=True)

import glob
print("=== REG METADATA (FORM=%s) ===" % FORM, flush=True)
seen = set()
for f in sorted(glob.glob(f"/tmp/dscan_{tag}/**/*", recursive=True)):
    if not os.path.isfile(f):
        continue
    try:
        txt = open(f, errors="ignore").read()
    except Exception:
        continue
    for line in txt.splitlines():
        ll = line.lower()
        if any(k in ll for k in ["vgpr_count", "agpr_count", "vgpr_spill", "agpr_spill",
                                 "next_free_vgpr", "next_free_agpr", ".vgpr", ".agpr",
                                 "scratch_size", "numvgpr", "numagpr"]):
            s = line.strip()
            if s not in seen:
                seen.add(s)
                print(f"  [{os.path.basename(f)}] {s}", flush=True)

