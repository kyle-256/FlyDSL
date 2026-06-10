"""Round-35: A/B test removing pipe s_setprio (TK_FP4_SETPRIO env, read at import).
Full 14-shape recommend_config-routed fly TFLOPS (geomean) + det0 on kv M4096
(race-history shape). Run twice (env=1 setprio-on baseline, env=0 setprio-off)."""
import os, sys, math, torch
_R = "/workspace/code/FlyDSL"
for p in (_R, _R + "/flydsl/src"):
    sys.path.insert(0, p)
import flydsl.compiler as flyc
from turbo.mxfp4_gemm_8wave import compile_mxfp4_gemm_8w, recommend_config, _PIPE_SETPRIO
from turbo.mxfp8_gemm_8wave import preshuffle_scale, preshuffle_scale_b_comb
SB = 32
print(f"=== TK_FP4_SETPRIO={os.environ.get('TK_FP4_SETPRIO','1')} -> _PIPE_SETPRIO={_PIPE_SETPRIO} ===", flush=True)


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


def ascale(asc, M, K, BM):
    nta = BM // 64; q = 16 * nta; pad = ((M + q - 1) // q) * q
    if pad != M:
        ap = torch.zeros((pad, K // SB), dtype=torch.uint8, device=asc.device); ap[:M] = asc; asc = ap
    return preshuffle_scale(asc, K, nta).view(-1)


def mk(M, N, K):
    d = "cuda"
    a = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device=d)
    b = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=d)
    asc = torch.randint(125, 130, (M, K // SB), dtype=torch.uint8, device=d)
    bsc = torch.randint(125, 130, (N, K // SB), dtype=torch.uint8, device=d)
    return a.view(torch.int8).view(-1), b.view(torch.int8).view(-1), asc, bsc


def one(M, N, K):
    ai, bi, asc, bsc = mk(M, N, K)
    BM, BN, gm, gn = recommend_config(M, N, K)
    asp = ascale(asc, M, K, BM)
    bsp = (preshuffle_scale_b_comb(bsc, K) if BN >= 256 else preshuffle_scale(bsc, K, BN // 128)).view(-1)
    c = torch.zeros((M, N), dtype=torch.bfloat16, device="cuda")
    st = torch.cuda.current_stream()
    fn = compile_mxfp4_gemm_8w(K=K, BLOCK_M=BM, BLOCK_N=BN, mode="pipe", group_m=gm, group_n=gn)
    ar = (ai, bi, c.view(-1), asp, bsp, M, N, st)
    cc = flyc.compile(fn, *ar); cc(*ar); torch.cuda.synchronize()
    tf = 2 * M * N * K / (bench(lambda: cc(*ar)) / 1e6) / 1e12
    return tf, cc, ar, c


SHAPES = [("7Bqo", 4096, 4096), ("7Bgu", 11008, 4096), ("7Bdn", 4096, 11008),
          ("70Bqo", 8192, 8192), ("70Bkv", 1024, 8192), ("70Bgu", 28672, 8192), ("70Bdn", 8192, 28672)]
lr = 0.0
for M in (4096, 8192):
    for tag, N, K in SHAPES:
        tf, *_ = one(M, N, K)
        lr += math.log(tf)
        print(f"  {tag:6s} M={M:5d} N={N:5d} K={K:5d}  {tf:6.0f} TF", flush=True)
print(f"GEOMEAN fly TFLOPS = {math.exp(lr/14):.1f}", flush=True)

# det0 on kv M4096 (race-history shape)
print("--- det0 300-run kv M4096 ---", flush=True)
M, N, K = 4096, 1024, 8192
fnm = compile_mxfp4_gemm_8w(K=K, **dict(zip(("BLOCK_M","BLOCK_N","group_m","group_n"), recommend_config(M,N,K))), mode="pipe")
maxd = 0.0; nan = False
for r in range(300):
    ai, bi, asc, bsc = mk(M, N, K)
    BM, BN, gm, gn = recommend_config(M, N, K)
    asp = ascale(asc, M, K, BM); bsp = (preshuffle_scale_b_comb(bsc, K) if BN>=256 else preshuffle_scale(bsc, K, BN//128)).view(-1)
    c1 = torch.zeros((M, N), dtype=torch.bfloat16, device="cuda"); st = torch.cuda.current_stream()
    ar = (ai, bi, c1.view(-1), asp, bsp, M, N, st)
    cc = flyc.compile(fnm, *ar); cc(*ar); torch.cuda.synchronize(); o1 = c1.clone()
    c1.zero_(); cc(*ar); torch.cuda.synchronize()
    maxd = max(maxd, (o1.float()-c1.float()).abs().max().item())
    if torch.isnan(c1).any(): nan = True
print(f"det0: 300-run maxdiff={maxd} nan={nan} -> {'DET0' if maxd==0 and not nan else 'FAIL'}", flush=True)
print("DONE", flush=True)
