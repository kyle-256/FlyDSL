"""Round-48: quantify pivot #2 (relax preshuffle-B). Bench aiter bpreshuffle=True
(competitor, B preshuffled) vs aiter bpreshuffle=False (B NOT preshuffled = OUR hard
constraint) vs fly, on bulk shapes. Tells the user: how much does preshuffle-B buy
aiter, and where does fly sit vs the SAME-CONSTRAINT SOTA (aiter-noBpre)? GPU7 only.
r43 only counted barriers in the two aiter .co; this measures their actual perf."""
import sys, os, torch
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
    for _ in range(wu): fn()
    torch.cuda.synchronize(); best = 1e9
    for _ in range(reps):
        e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(); e0.record()
        for _ in range(it): fn()
        e1.record(); torch.cuda.synchronize(); best = min(best, e0.elapsed_time(e1) * 1000 / it)
    return best


def tf(M, N, K, us):
    return 2 * M * N * K / (us / 1e6) / 1e12


def aiter_variants(M, N, K):
    x = torch.randn((M, K), dtype=torch.bfloat16, device="cuda")
    w = torch.randn((N, K), dtype=torch.bfloat16, device="cuda")
    xq, xs = quant_func(x, shuffle=True)
    wq, ws = quant_func(w, shuffle=True)
    wsh = shuffle_weight(wq, layout=(16, 16))
    out = {}
    # competitor (B preshuffled)
    try:
        fb = lambda: aiter.gemm_a4w4(xq, wsh, xs, ws, bpreshuffle=True)
        ob = fb(); torch.cuda.synchronize()
        out["bpre"] = (tf(M, N, K, bench(fb)), ob)
    except Exception as e:
        out["bpre"] = (f"ERR:{type(e).__name__}", None)
    # same-constraint (B NOT preshuffled)
    try:
        fn = lambda: aiter.gemm_a4w4(xq, wq, xs, ws, bpreshuffle=False)
        on = fn(); torch.cuda.synchronize()
        out["nobpre"] = (tf(M, N, K, bench(fn)), on)
    except Exception as e:
        out["nobpre"] = (f"ERR:{type(e).__name__}", None)
    return out


def fly_tf(M, N, K):
    d = "cuda"
    BM, BN, gm, gn = recommend_config(M, N, K)
    a = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device=d)
    b = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=d)
    asc = torch.randint(125, 130, (M, K // SB), dtype=torch.uint8, device=d)
    bsc = torch.randint(125, 130, (N, K // SB), dtype=torch.uint8, device=d)
    c = torch.zeros((M, N), dtype=torch.bfloat16, device=d)
    nta = BM // 64; q = 16 * nta; pad = ((M + q - 1) // q) * q
    if pad != M:
        ap = torch.zeros((pad, K // SB), dtype=torch.uint8, device=d); ap[:M] = asc; asc = ap
    asp = preshuffle_scale(asc, K, nta)
    bsp = preshuffle_scale_b_comb(bsc, K) if BN >= 256 else preshuffle_scale(bsc, K, BN // 128)
    ai = a.view(torch.int8).view(-1); bi = b.view(torch.int8).view(-1); st = torch.cuda.current_stream()
    fn = compile_mxfp4_gemm_8w(K=K, BLOCK_M=BM, BLOCK_N=BN, mode="pipe", group_m=gm, group_n=gn)
    ar = (ai, bi, c.view(-1), asp.view(-1), bsp.view(-1), M, N, st)
    cc = flyc.compile(fn, *ar); cc(*ar); torch.cuda.synchronize()
    return tf(M, N, K, bench(lambda: cc(*ar)))


# bulk shapes (where preshuffle-B matters most + noBpre 256x256 applies)
SHAPES = [("7B q/o", 4096, 4096), ("70B q/o", 8192, 8192), ("70B down", 8192, 28672)]
print(f"{'shape':12s}{'M':>6}{'N':>6}{'K':>6}  {'aiBpre':>7s}{'aiNoBpre':>9s}{'fly':>7s}  {'preB-gain':>9s}{'fly/noBpre':>11s}{'fly/Bpre':>9s}  sane")
for M in (4096, 8192):
    for tag, N, K in SHAPES:
        av = aiter_variants(M, N, K)
        tb, ob = av["bpre"]; tn, on = av["nobpre"]
        ft = fly_tf(M, N, K)
        sane = "n/a"
        if ob is not None and on is not None:
            sane = "ok" if torch.allclose(ob.float(), on.float(), atol=1e-1, rtol=1e-1) else "MISMATCH"
        if isinstance(tb, float) and isinstance(tn, float):
            print(f"{tag:12s}{M:>6}{N:>6}{K:>6}  {tb:>7.0f}{tn:>9.0f}{ft:>7.0f}  {tb/tn:>9.3f}{ft/tn:>11.3f}{ft/tb:>9.3f}  {sane}")
        else:
            print(f"{tag:12s}{M:>6}{N:>6}{K:>6}  Bpre={tb} noBpre={tn} fly={ft:.0f}")
