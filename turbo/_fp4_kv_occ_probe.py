"""Round-16 (Mode-B B3 r3_1 feasibility spike): kv occupancy headroom.

Question: is kv M4096 (BN128 = 128 wg, grid-underfilled vs 256 CUs) still
grid-occupancy-bound? If so, BN=64 (-> 256 wg) is worth a multi-session quadrant
rewrite. If the per-FLOP TFLOPS curve flattens at <=128 wg, BN=64 has no headroom
and B3 is dead (avoids wasting sessions on a useless rewrite).

Method: fix N=1024 K=8192 BN=128, sweep M so grid (= ceil(M/256)*ceil(N/128))
goes 64 -> 128 -> 256 -> 512 wg. Per-FLOP TFLOPS vs grid-size = the occupancy
curve. Inflection point tells us where 256 CUs saturate. Also report BN256 and
aiter for context. No kernel change (pure perf probe)."""
import sys, os, torch
_R = "/workspace/code/FlyDSL"
for p in (_R, _R + "/flydsl/src"):
    sys.path.insert(0, p)
import aiter
from aiter.ops.shuffle import shuffle_weight
import flydsl.compiler as flyc
from turbo.mxfp4_gemm_8wave import compile_mxfp4_gemm_8w
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


def mine_tf(M, N, K, BN):
    d = "cuda"
    a = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device=d)
    b = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=d)
    asc = torch.randint(125, 130, (M, K // SB), dtype=torch.uint8, device=d)
    bsc = torch.randint(125, 130, (N, K // SB), dtype=torch.uint8, device=d)
    c = torch.zeros((M, N), dtype=torch.bfloat16, device=d)
    asp = preshuffle_scale(asc, K, 4)
    bsp = preshuffle_scale_b_comb(bsc, K)
    ai = a.view(torch.int8).view(-1)
    bi = b.view(torch.int8).view(-1)
    st = torch.cuda.current_stream()
    fn = compile_mxfp4_gemm_8w(K=K, BLOCK_M=256, BLOCK_N=BN, mode="pipe", group_m=4, group_n=0)
    ar = (ai, bi, c.view(-1), asp.view(-1), bsp.view(-1), M, N, st)
    cc = flyc.compile(fn, *ar); cc(*ar); torch.cuda.synchronize()
    return 2 * M * N * K / (bench(lambda: cc(*ar)) / 1e6) / 1e12


N, K = 1024, 8192
print(f"kv occupancy headroom: N={N} K={K}, per-FLOP TFLOPS vs grid size")
print(f"{'M':>6} {'wg256':>6} {'wg128':>6} {'fly256':>8} {'fly128':>8} {'aiter':>8} {'f128/ai':>8}")
for M in (2048, 4096, 8192, 16384):
    wg256 = ((M + 255) // 256) * ((N + 255) // 256)
    wg128 = ((M + 255) // 256) * ((N + 127) // 128)
    t256 = mine_tf(M, N, K, 256)
    t128 = mine_tf(M, N, K, 128)
    at = comp_tf(M, N, K)
    print(f"{M:>6} {wg256:>6} {wg128:>6} {t256:>8.0f} {t128:>8.0f} {at:>8.0f} {t128/at:>8.3f}")
    sys.stdout.flush()
