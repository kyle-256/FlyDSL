"""Round-38: per-shape num_xcds sweep (XCD->CU remap, gemm-opt S13 L2-locality lever).
num_xcds is hardwired 8 in recommend_config/launch and was NEVER per-shape swept with
the reliable interleaved protocol (r26-28 covered autotune/group_m/group_n only).
Pure tile->CU permutation = bit-exact (output identical), so correctness is trivial;
this is a config-only Mode-A lever. Interleaved A/B (alternate nx per trial, win-count
vs nx=8 anchor) on representative shapes. nx in {4,8,16}."""
import sys, torch
_R = "/workspace/code/FlyDSL"
for p in (_R, _R + "/flydsl/src"):
    sys.path.insert(0, p)
import flydsl.compiler as flyc
from turbo.mxfp4_gemm_8wave import compile_mxfp4_gemm_8w, recommend_config
from turbo.mxfp8_gemm_8wave import preshuffle_scale, preshuffle_scale_b_comb
SB = 32


def snr(o, r):
    o = o.float(); r = r.float()
    return 10 * torch.log10((r ** 2).sum() / ((o - r) ** 2).sum().clamp_min(1e-20)).item()


def bench(fn, it=30, wu=8, reps=2):
    fn(); torch.cuda.synchronize()
    for _ in range(wu):
        fn()
    torch.cuda.synchronize(); best = 1e9
    for _ in range(reps):
        e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(); e0.record()
        for _ in range(it):
            fn()
        e1.record(); torch.cuda.synchronize(); best = min(best, e0.elapsed_time(e1) * 1000 / it)
    return best


def ascale(asc, M, K, BM):
    nta = BM // 64; q = 16 * nta; pad = ((M + q - 1) // q) * q
    if pad != M:
        ap = torch.zeros((pad, K // SB), dtype=torch.uint8, device=asc.device); ap[:M] = asc; asc = ap
    return preshuffle_scale(asc, K, nta).view(-1)


def build(M, N, K, nx):
    d = "cuda"
    a = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device=d)
    b = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=d)
    asc = torch.randint(125, 130, (M, K // SB), dtype=torch.uint8, device=d)
    bsc = torch.randint(125, 130, (N, K // SB), dtype=torch.uint8, device=d)
    BM, BN, gm, gn = recommend_config(M, N, K)
    asp = ascale(asc, M, K, BM)
    bsp = (preshuffle_scale_b_comb(bsc, K) if BN >= 256 else preshuffle_scale(bsc, K, BN // 128)).view(-1)
    c = torch.zeros((M, N), dtype=torch.bfloat16, device=d)
    st = torch.cuda.current_stream()
    fn = compile_mxfp4_gemm_8w(K=K, BLOCK_M=BM, BLOCK_N=BN, mode="pipe", group_m=gm, group_n=gn, num_xcds=nx)
    ar = (a.view(torch.int8).view(-1), b.view(torch.int8).view(-1), c.view(-1), asp, bsp, M, N, st)
    cc = flyc.compile(fn, *ar); cc(*ar); torch.cuda.synchronize()
    return cc, ar, c, (BM, BN, gm, gn)


REP = [("70Bdn", 8192, 8192, 28672), ("70Bgu", 8192, 28672, 8192),
       ("70Bqo", 8192, 8192, 8192), ("70Bkv", 4096, 1024, 8192)]
NX = [4, 8, 16]
T = 8
print("shape    cfg              | " + "  ".join(f"nx{n:<2d}" for n in NX) + "  | wins vs nx8 | SNR(nx4,nx16 vs nx8)")
for tag, M, N, K in REP:
    builds = {}
    for nx in NX:
        builds[nx] = build(M, N, K, nx)
    cfg = builds[8][3]
    # correctness: bit-exact vs nx8 (pure permutation)
    ref = builds[8][2].clone()
    s4 = snr(builds[4][2], ref); s16 = snr(builds[16][2], ref)
    # interleaved bench
    tf = {n: [] for n in NX}
    for t in range(T):
        for nx in NX:
            cc, ar, c, _ = builds[nx]
            us = bench(lambda: cc(*ar))
            tf[nx].append(2 * M * N * K / (us / 1e6) / 1e12)
    import statistics as st_
    med = {n: st_.median(tf[n]) for n in NX}
    wins4 = sum(1 for i in range(T) if tf[4][i] > tf[8][i])
    wins16 = sum(1 for i in range(T) if tf[16][i] > tf[8][i])
    print(f"{tag:7s} {str(cfg):16s} | " + "  ".join(f"{med[n]:5.0f}" for n in NX)
          + f"  | nx4 {wins4}/{T} nx16 {wins16}/{T} | {s4:.0f},{s16:.0f} dB", flush=True)
print("DONE", flush=True)
