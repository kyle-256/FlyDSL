"""round-56: re-verify num_xcds for the NEW BM128/gm8 kv config (r54 BM192->BM128,
r55 gm2->gm8). r38 confirmed nx8 optimal but on the OLD grid (BM192/BM256). The
XCD->CU remap (grouped_xcd_pid) interacts with both BLOCK_M (M-tile count) and
group_m (super-block width), both of which changed for kv M4096. So re-verify the
kv-specific num_xcds on the new BM128/gm8 grid (the logical completion of the
BM128 kv config tuning). Pure XCD->CU bijection = bit-exact / det-neutral.

Shared inputs across nx (proper bit-exact check, fixing r38's fresh-randint SNR
artifact). Interleaved A/B (thermal-matched) + win-count vs nx8 anchor on both kv
shapes (M4096 = binding min_ratio, BM128/gm8; M8192 = BM256/gm2). A reliable >=1%
winner (>=8/10) routes; else ROLLBACK (nx8 stays, dimension closed for new config).
"""
import sys, torch, statistics as st_
_R = "/workspace/code/FlyDSL"
for p in (_R, _R + "/flydsl/src"):
    sys.path.insert(0, p)
import flydsl.compiler as flyc
from turbo.mxfp4_gemm_8wave import compile_mxfp4_gemm_8w, recommend_config
from turbo.mxfp8_gemm_8wave import preshuffle_scale, preshuffle_scale_b_comb
SB = 32


def mkraw(M, N, K):
    d = "cuda"
    a = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device=d)
    b = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=d)
    asc = torch.randint(125, 130, (M, K // SB), dtype=torch.uint8, device=d)
    bsc = torch.randint(125, 130, (N, K // SB), dtype=torch.uint8, device=d)
    return a.view(torch.int8).view(-1), b.view(torch.int8).view(-1), asc, bsc


def ascale(asc, M, K, BM):
    nta = BM // 64; q = 16 * nta; pad = ((M + q - 1) // q) * q
    if pad != M:
        ap = torch.zeros((pad, K // SB), dtype=torch.uint8, device=asc.device); ap[:M] = asc; asc = ap
    return preshuffle_scale(asc, K, nta).view(-1)


def build(M, N, K, a, b, asc, bsc, nx):
    BM, BN, gm, gn = recommend_config(M, N, K)
    asp = ascale(asc, M, K, BM)
    bsp = (preshuffle_scale_b_comb(bsc, K) if BN >= 256 else preshuffle_scale(bsc, K, BN // 128)).view(-1)
    c = torch.zeros((M, N), dtype=torch.bfloat16, device="cuda")
    st = torch.cuda.current_stream()
    fn = compile_mxfp4_gemm_8w(K=K, BLOCK_M=BM, BLOCK_N=BN, mode="pipe", group_m=gm, group_n=gn, num_xcds=nx)
    ar = (a, b, c.view(-1), asp, bsp, M, N, st)
    cc = flyc.compile(fn, *ar); cc(*ar); torch.cuda.synchronize()
    return cc, ar, c, (BM, BN, gm, gn)


def bench(cc, ar, it=30, wu=10, reps=2):
    for _ in range(wu):
        cc(*ar)
    torch.cuda.synchronize(); best = 1e9
    for _ in range(reps):
        e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(); e0.record()
        for _ in range(it):
            cc(*ar)
        e1.record(); torch.cuda.synchronize(); best = min(best, e0.elapsed_time(e1) * 1000 / it)
    return best


NX = [4, 8, 16]
T = 10
for tag, M, N, K in [("70Bkv-M4096", 4096, 1024, 8192), ("70Bkv-M8192", 8192, 1024, 8192)]:
    a, b, asc, bsc = mkraw(M, N, K)
    builds = {nx: build(M, N, K, a, b, asc, bsc, nx) for nx in NX}
    cfg = builds[8][3]
    base = builds[8][2].clone()
    md = {nx: (builds[nx][2].float() - base.float()).abs().max().item() for nx in (4, 16)}
    print(f"{tag} cfg(BM,BN,gm,gn)={cfg} bit-diff nx4={md[4]:.4g} nx16={md[16]:.4g} (bijection=>0)", flush=True)
    tf = {nx: [] for nx in NX}
    for t in range(T):
        for nx in NX:
            cc, ar, _, _ = builds[nx]
            us = bench(cc, ar)
            tf[nx].append(2 * M * N * K / (us / 1e6) / 1e12)
    med = {nx: st_.median(tf[nx]) for nx in NX}
    w4 = sum(1 for i in range(T) if tf[4][i] > tf[8][i])
    w16 = sum(1 for i in range(T) if tf[16][i] > tf[8][i])
    print(f"  nx4={med[4]:.0f}TF nx8={med[8]:.0f}TF nx16={med[16]:.0f}TF | "
          f"nx4/nx8={med[4]/med[8]:.3f} ({w4}/{T})  nx16/nx8={med[16]/med[8]:.3f} ({w16}/{T})", flush=True)
print("DONE", flush=True)
