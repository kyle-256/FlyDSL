"""round-89 (Rule-7 L3 ISA register-budget inspection): r87/r88 refuted the
1-stage `staged` and BLOCK_N=128 variants on PERF but never measured their actual
VGPR/AGPR/spill.  This quantifies the occupancy-avenue closure with hard ISA
register numbers for the 3 variants on one bulk shape (70B q/o 8192^3):
  (a) pipe / BN256  = production (r85: V=228, A=0, occ 5.46%)
  (b) staged / BN256 = 1-stage (r87 refuted, 0.75-0.84x) — should drop next-frags
  (c) pipe / BN128   = half-N-tile (r88 refuted, 0.81-0.89x) — should halve acc VGPR
If (b)/(c) DO drop VGPR yet still lost on perf, it definitively proves the
occupancy gain is real but insufficient (tile/stage shrink cost dominates) — no
kernel-source VGPR lever converts occupancy into speed.  Dump-only, no GPU loop."""
import sys, os, glob, re
_R = "/workspace/code/FlyDSL"
for p in (_R, _R + "/flydsl/src"):
    sys.path.insert(0, p)


def dump_one(tag, BN, mode, M=8192, N=8192, K=8192):
    d = f"/tmp/isa_{tag}"
    os.system(f"rm -rf {d}")
    os.environ["FLYDSL_DUMP_IR"] = "1"
    os.environ["FLYDSL_DUMP_DIR"] = d
    os.environ["FLYDSL_RUNTIME_ENABLE_CACHE"] = "0"
    import torch
    import flydsl.compiler as flyc
    from turbo.mxfp4_gemm_8wave import compile_mxfp4_gemm_8w
    from turbo.mxfp8_gemm_8wave import preshuffle_scale, preshuffle_scale_b_comb
    SB = 32
    dev = "cuda"
    a = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device=dev)
    b = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=dev)
    asc = torch.randint(125, 130, (M, K // SB), dtype=torch.uint8, device=dev)
    bsc = torch.randint(125, 130, (N, K // SB), dtype=torch.uint8, device=dev)
    c = torch.zeros((M, N), dtype=torch.bfloat16, device=dev)
    asp = preshuffle_scale(asc, K, 256 // 64).view(-1)
    bsp = (preshuffle_scale_b_comb(bsc, K) if BN >= 256 else preshuffle_scale(bsc, K, BN // 128)).view(-1)
    st = torch.cuda.current_stream()
    fn = compile_mxfp4_gemm_8w(K=K, BLOCK_M=256, BLOCK_N=BN, mode=mode, group_m=4, group_n=0, num_xcds=8)
    ar = (a.view(torch.int8).view(-1), b.view(torch.int8).view(-1), c.view(-1), asp, bsp, M, N, st)
    cc = flyc.compile(fn, *ar); cc(*ar); torch.cuda.synchronize()
    # find the final ISA
    cands = glob.glob(f"{d}/**/*final_isa.s", recursive=True) + glob.glob(f"{d}/**/*.s", recursive=True)
    if not cands:
        print(f"{tag:18s}: NO ISA dumped (dirs: {os.listdir(d) if os.path.isdir(d) else 'none'})")
        return
    s = max(cands, key=os.path.getsize)
    txt = open(s).read()
    def g(pat):
        m = re.search(pat, txt)
        return m.group(1) if m else "?"
    v = g(r"\.vgpr_count:\s*(\d+)")
    a_ = g(r"\.agpr_count:\s*(\d+)")
    sp = g(r"\.vgpr_spill_count:\s*(\d+)")
    nf = g(r"next_free_vgpr:?\s*(\d+)")
    sg = g(r"\.sgpr_count:\s*(\d+)")
    nbar = txt.count("s_barrier")
    nmfma = len(re.findall(r"v_mfma", txt))
    print(f"{tag:18s}: vgpr={v} agpr={a_} spill={sp} next_free_vgpr={nf} sgpr={sg} | s_barrier={nbar} v_mfma={nmfma} | {os.path.basename(s)}")


print("== round-89 ISA register-budget (70B q/o 8192^3) ==")
dump_one("pipe_BN256(prod)", 256, "pipe")
dump_one("staged_BN256", 256, "staged")
dump_one("pipe_BN128", 128, "pipe")
