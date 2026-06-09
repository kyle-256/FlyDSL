"""Round-33 re-baseline: B6-landed production (recommend_config-routed) FlyDSL mxfp4
vs aiter gemm_a4w4, full 14-shape Llama 7B/70B (M in {4096,8192}). Reports per-shape
fly/aiter + geomean + min_ratio. 'mine' uses recommend_config (kv BN128/gm2/BM192,
big-N gn14) — the actual production path, unlike _fp4_vs_aiter.py which is fixed
BM256/BN256/gm4. Measurement-only (no kernel change)."""
import sys, math, torch
_R = "/workspace/code/FlyDSL"
for p in (_R, _R + "/flydsl/src"):
    sys.path.insert(0, p)
import aiter
from aiter.ops.shuffle import shuffle_weight
import flydsl.compiler as flyc
from turbo.mxfp4_gemm_8wave import compile_mxfp4_gemm_8w, recommend_config
from turbo.mxfp8_gemm_8wave import preshuffle_scale, preshuffle_scale_b_comb
SB = 32
quant_func = aiter.get_triton_quant(aiter.QuantType.per_1x32)


def bench(fn, it=30, wu=10, reps=4):
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


def comp_tf(M, N, K):
    x = torch.randn((M, K), dtype=torch.bfloat16, device="cuda")
    w = torch.randn((N, K), dtype=torch.bfloat16, device="cuda")
    xq, xs = quant_func(x, shuffle=True)
    wq, ws = quant_func(w, shuffle=True)
    wsh = shuffle_weight(wq, layout=(16, 16))
    fn = lambda: aiter.gemm_a4w4(xq, wsh, xs, ws, bpreshuffle=True)
    fn()
    return 2 * M * N * K / (bench(fn) / 1e6) / 1e12


def ascale(asc, M, K, BM):
    nta = BM // 64
    q = 16 * nta
    pad = ((M + q - 1) // q) * q
    if pad != M:
        ap = torch.zeros((pad, K // SB), dtype=torch.uint8, device=asc.device)
        ap[:M] = asc
        asc = ap
    return preshuffle_scale(asc, K, nta).view(-1)


def mine_tf(M, N, K):
    d = "cuda"
    a = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device=d)
    b = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=d)
    asc = torch.randint(125, 130, (M, K // SB), dtype=torch.uint8, device=d)
    bsc = torch.randint(125, 130, (N, K // SB), dtype=torch.uint8, device=d)
    BM, BN, gm, gn, nx = recommend_config(M, N, K)
    asp = ascale(asc, M, K, BM)
    bsp = (preshuffle_scale_b_comb(bsc, K) if BN >= 256 else preshuffle_scale(bsc, K, BN // 128)).view(-1)
    c = torch.zeros((M, N), dtype=torch.bfloat16, device=d)
    st = torch.cuda.current_stream()
    fn = compile_mxfp4_gemm_8w(K=K, BLOCK_M=BM, BLOCK_N=BN, mode="pipe", group_m=gm, group_n=gn, num_xcds=nx)
    ar = (a.view(torch.int8).view(-1), b.view(torch.int8).view(-1), c.view(-1), asp, bsp, M, N, st)
    cc = flyc.compile(fn, *ar); cc(*ar); torch.cuda.synchronize()
    return 2 * M * N * K / (bench(lambda: cc(*ar)) / 1e6) / 1e12, (BM, BN, gm, gn, nx)


SHAPES = [
    ("7B q/o", 4096, 4096), ("7B gate/up", 11008, 4096), ("7B down", 4096, 11008),
    ("70B q/o", 8192, 8192), ("70B kv", 1024, 8192),
    ("70B gate/up", 28672, 8192), ("70B down", 8192, 28672),
]
print(f"{'shape':12s} {'M':>5s} {'N':>5s} {'K':>5s}  {'aiter':>6s} {'fly':>6s}  {'cfg':>14s}  fly/aiter")
lr = 0.0; mn = 1e9; npass = 0
for M in (4096, 8192):
    print(f"--- M={M} ---")
    for tag, N, K in SHAPES:
        ct = comp_tf(M, N, K)
        ft, cfg = mine_tf(M, N, K)
        r = ft / ct
        lr += math.log(r); mn = min(mn, r); npass += (r >= 0.96)
        print(f"{tag:12s} {M:5d} {N:5d} {K:5d}  {ct:6.0f} {ft:6.0f}  {str(cfg):>14s}  {r:.3f}")
print(f"GEOMEAN fly/aiter = {math.exp(lr/14):.3f}  min_ratio = {mn:.3f}  shapes>=0.96: {npass}/14")
