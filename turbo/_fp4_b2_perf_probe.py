"""B2 r_b1 perf check: asm-AGPR (agpr=128, V+A=384 -> 1 wave/SIMD) vs intrinsic
(V=234 -> 2 waves) on bulk shapes. Confirms whether asm-AGPR-inplace alone helps
bulk despite the occupancy halving (V was NOT freed). Hot best-of-3."""
import sys, torch
_R = "/workspace/code/FlyDSL"
for p in (_R, _R + "/flydsl/src"):
    sys.path.insert(0, p)
import flydsl.compiler as flyc
from turbo.mxfp4_gemm_8wave import compile_mxfp4_gemm_8w
from turbo.mxfp8_gemm_8wave import preshuffle_scale, preshuffle_scale_b_comb
SB = 32


def bench(fn, it=30, wu=8, reps=3):
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


def one(M, N, K, asm_mfma, agpr):
    d = "cuda"
    a = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device=d)
    b = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=d)
    asc = torch.randint(125, 130, (M, K // SB), dtype=torch.uint8, device=d)
    bsc = torch.randint(125, 130, (N, K // SB), dtype=torch.uint8, device=d)
    asp = preshuffle_scale(asc, K, 256 // 64).view(-1)
    bsp = preshuffle_scale_b_comb(bsc, K).view(-1)
    c = torch.zeros((M, N), dtype=torch.bfloat16, device=d)
    fn = compile_mxfp4_gemm_8w(K=K, BLOCK_M=256, BLOCK_N=256, mode="pipe", wave_topo="2x4",
                               asm_mfma=asm_mfma, agpr_alloc=agpr, group_m=4, group_n=0)
    ar = (a.view(torch.int8).view(-1), b.view(torch.int8).view(-1), c.view(-1), asp, bsp, M, N, torch.cuda.current_stream())
    cc = flyc.compile(fn, *ar)
    us = bench(lambda: cc(*ar))
    return 2 * M * N * K / (us / 1e6) / 1e12


for tag, M, N, K in [("70B q/o square", 8192, 8192, 8192), ("70B down big-K", 8192, 8192, 28672)]:
    tf_i = one(M, N, K, False, 0)     # intrinsic baseline
    tf_a = one(M, N, K, True, 128)    # asm-AGPR
    print(f"{tag} M={M} N={N} K={K}: intrinsic={tf_i:.0f} TF  asm-AGPR={tf_a:.0f} TF  win={(tf_a/tf_i-1)*100:+.1f}%")
