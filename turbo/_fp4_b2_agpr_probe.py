"""B2 r_b1: does the fp8-case asm-AGPR mechanism (amdgpu-agpr-alloc + mfma-vgpr-form
passthrough FUNCTION attr + asm_mma mode2 `=a`) move the mxfp4 accumulator into
AGPR and FREE the VGPR file (V<228)? round-14 used an asm-constraint pin (wrong)
and got V=256 + spill 4485. This uses the correct passthrough mechanism.

For each variant: compile pipe (BN256), run (SNR correctness), dump ISA, grep
.vgpr_count / .agpr_count / .vgpr_spill_count from amdhsa.kernels metadata.
Tiny shape (correctness + register alloc is shape-independent; avoids big-spill crash)."""
import os, sys, glob, re, torch
_R = "/workspace/code/FlyDSL"
for p in (_R, _R + "/flydsl/src"):
    sys.path.insert(0, p)
SB = 32


def ref_gemm(a_u8, b_u8, asc, bsc, M, N, K, fp4_utils):
    a_f = fp4_utils.mxfp4_to_f32(a_u8)[:M, :K].float()
    b_f = fp4_utils.mxfp4_to_f32(b_u8)[:N, :K].float()
    a_s = fp4_utils.e8m0_to_f32(asc).repeat_interleave(SB, dim=-1)[:M, :K].float()
    b_s = fp4_utils.e8m0_to_f32(bsc).repeat_interleave(SB, dim=-1)[:N, :K].float()
    return torch.matmul(a_f * a_s, (b_f * b_s).T)


def snr(o, r):
    o, r = o.float(), r.float()
    n = (o - r).pow(2).mean()
    return float("inf") if n.item() == 0 else (10 * torch.log10(r.pow(2).mean() / n)).item()


def grep_regs(dump_dir):
    files = glob.glob(f"{dump_dir}/**/*final_isa.s", recursive=True)
    if not files:
        return "no .s found"
    txt = open(files[0]).read()
    out = {}
    for key in ("vgpr_count", "agpr_count", "vgpr_spill_count", "sgpr_spill_count"):
        m = re.search(rf"\.{key}:\s*(\d+)", txt)
        out[key] = m.group(1) if m else "?"
    nspill = txt.count("scratch_store") + txt.count("scratch_load")
    return f"V={out['vgpr_count']} A={out['agpr_count']} vspill={out['vgpr_spill_count']} scratch_ops={nspill}"


def run(label, asm_mfma, agpr):
    M, N, K, BM, BN = 256, 256, 512, 256, 256
    dump = f"/tmp/b2dump_{label}"
    os.system(f"rm -rf {dump}")
    os.environ["FLYDSL_DUMP_IR"] = "1"
    os.environ["FLYDSL_DUMP_DIR"] = dump
    os.environ["FLYDSL_RUNTIME_ENABLE_CACHE"] = "0"
    import importlib, flydsl.compiler as flyc
    from tests.kernels.utils import fp4_utils
    from turbo.mxfp4_gemm_8wave import compile_mxfp4_gemm_8w
    from turbo.mxfp8_gemm_8wave import preshuffle_scale, preshuffle_scale_b_comb
    d = "cuda"
    a = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device=d)
    b = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=d)
    asc = torch.randint(125, 130, (M, K // SB), dtype=torch.uint8, device=d)
    bsc = torch.randint(125, 130, (N, K // SB), dtype=torch.uint8, device=d)
    ref = ref_gemm(a, b, asc, bsc, M, N, K, fp4_utils)
    asp = preshuffle_scale(asc, K, BM // 64).view(-1)
    bsp = preshuffle_scale_b_comb(bsc, K).view(-1)
    c = torch.zeros((M, N), dtype=torch.bfloat16, device=d)
    fn = compile_mxfp4_gemm_8w(K=K, BLOCK_M=BM, BLOCK_N=BN, mode="pipe", wave_topo="2x4",
                               asm_mfma=asm_mfma, agpr_alloc=agpr, group_m=4, group_n=0)
    ar = (a.view(torch.int8).view(-1), b.view(torch.int8).view(-1), c.view(-1), asp, bsp, M, N, torch.cuda.current_stream())
    cc = flyc.compile(fn, *ar); cc(*ar); torch.cuda.synchronize()
    s = snr(c.float(), ref)
    print(f"[{label}] asm_mfma={asm_mfma} agpr_alloc={agpr}  SNR={s:6.2f} dB  {grep_regs(dump)}")


if __name__ == "__main__":
    run("baseline_intrinsic", False, 0)
    run("b2_asm_agpr128", True, 128)
