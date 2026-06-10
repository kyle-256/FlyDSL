"""B4 r4_3 perf probe: 4x2-BN64 pipe vs production 2x4-BN128 pipe on kv shapes.
Hot best-of-3 (warm), confirms the kv occupancy lever before r4_4 routing."""
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


def one(M, N, K, BM, BN, topo):
    d = "cuda"
    a = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device=d)
    b = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=d)
    asc = torch.randint(125, 130, (M, K // SB), dtype=torch.uint8, device=d)
    bsc = torch.randint(125, 130, (N, K // SB), dtype=torch.uint8, device=d)
    n_ta = BM // 128 if topo == "4x2" else BM // 64
    n_tb = BN // 64 if topo == "4x2" else BN // 128
    asp = preshuffle_scale(asc, K, n_ta).view(-1)
    bsp = (preshuffle_scale_b_comb(bsc, K) if BN >= 256 else preshuffle_scale(bsc, K, n_tb)).view(-1)
    c = torch.zeros((M, N), dtype=torch.bfloat16, device=d)
    fn = compile_mxfp4_gemm_8w(K=K, BLOCK_M=BM, BLOCK_N=BN, mode="pipe", wave_topo=topo, group_m=4, group_n=0)
    ar = (a.view(torch.int8).view(-1), b.view(torch.int8).view(-1), c.view(-1), asp, bsp, M, N, torch.cuda.current_stream())
    cc = flyc.compile(fn, *ar)
    us = bench(lambda: cc(*ar))
    return 2 * M * N * K / (us / 1e6) / 1e12


for M in (4096, 8192):
    K = 8192; N = 1024
    tf128 = one(M, N, K, 256, 128, "2x4")  # production kv
    tf64 = one(M, N, K, 256, 64, "4x2")    # B4
    print(f"kv M={M} N={N} K={K}: 2x4-BN128={tf128:.0f} TF  4x2-BN64={tf64:.0f} TF  win={(tf64/tf128-1)*100:+.1f}%")
