"""Round-29 (Rule7 level-2 re-profile): kv occupancy-headroom curve. Confirm
(a) kv M4096 is occupancy-bound, (b) the occupancy ceiling at the WG-count a
BM192 kernel would hit at M4096 (176 active CU), to ground the Mode-B B6 GO/NO.

Production kv kernel = BM256 BN128 gm2 (round-26). Sweep M so WG = ceil(M/256)*8
spans {128,160,176,192,256} CU. Hot best-of-3. TFLOPS is per the M actually run
(so compare the slope, not absolute). The key signal: does TFLOPS keep rising
with WG in the underfill region (occupancy-bound) and what is the ~176wg level
(BM192@M4096 equiv occupancy, but with BM256's higher per-WG efficiency = an
UPPER bound on BM192@M4096)."""
import sys, torch
_R = "/workspace/code/FlyDSL"
for p in (_R, _R + "/flydsl/src"):
    sys.path.insert(0, p)
import flydsl.compiler as flyc
from turbo.mxfp4_gemm_8wave import compile_mxfp4_gemm_8w
from turbo.mxfp8_gemm_8wave import preshuffle_scale
SB = 32
N, K, BN, GM = 1024, 8192, 128, 2


def make(M):
    d = "cuda"
    a = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device=d).view(torch.int8).view(-1)
    b = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=d).view(torch.int8).view(-1)
    asc = torch.randint(125, 130, (M, K // SB), dtype=torch.uint8, device=d)
    bsc = torch.randint(125, 130, (N, K // SB), dtype=torch.uint8, device=d)
    asp = preshuffle_scale(asc, K, 4).view(-1)
    bsp = preshuffle_scale(bsc, K, BN // 128).view(-1)
    return a, b, asp, bsp


def bench(fn, it=40, wu=12, reps=3):
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


fn = compile_mxfp4_gemm_8w(K=K, BLOCK_M=256, BLOCK_N=BN, mode="pipe", group_m=GM, group_n=0)
print(f"=== kv occupancy curve (BM256 BN128 gm2, N={N} K={K}) ===", flush=True)
print(f"{'M':>6s} {'WG':>4s} {'us':>7s} {'TFLOPS':>7s}", flush=True)
for M in (4096, 5120, 5632, 6144, 8192):
    a, b, asp, bsp = make(M)
    c = torch.zeros((M, N), dtype=torch.bfloat16, device="cuda")
    st = torch.cuda.current_stream()
    ar = (a, b, c.view(-1), asp, bsp, M, N, st)
    cc = flyc.compile(fn, *ar)
    us = bench(lambda: cc(*ar))
    wg = ((M + 255) // 256) * ((N + BN - 1) // BN)
    tf = 2 * M * N * K / (us / 1e6) / 1e12
    print(f"{M:6d} {wg:4d} {us:7.1f} {tf:7.0f}", flush=True)
print("DONE", flush=True)
