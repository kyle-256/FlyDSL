"""Bench aiter.gemm_a4w4 (no B-preshuffle) on the same harness/shape as test_mxfp4_4w,
for a fair fly-vs-aiter 对标. Same torch.cuda.Event 100-sample min/med timing.
Usage: python turbo/bench_aiter.py [M N K] [pre]   (pre=1 -> bpreshuffle=True)"""
import sys, torch, statistics
import aiter
try:
    import primus_turbo  # registers torch.ops.primus_turbo_cpp_extension (shuffle_scale/weight)
except Exception as _e:
    print("warn: primus_turbo import:", _e)

M, N, K = (int(x) for x in (sys.argv[1:4] or [8192, 8192, 28672]))
PRE = len(sys.argv) > 4 and sys.argv[4] == "1"
SB = 32
d = "cuda"; torch.manual_seed(0)
# fp4 packed (2 vals/byte) + e8m0 scales, same gen as test_mxfp4_4w
a = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device=d)
b = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=d)
asc = torch.randint(125, 130, (M, K // SB), dtype=torch.uint8, device=d)
bsc = torch.randint(125, 130, (N, K // SB), dtype=torch.uint8, device=d)

# aiter wants the fp4 packed dtype + e8m0 scale dtype (scale format perf-irrelevant).
af = a.view(torch.float4_e2m1fn_x2); bf = b.view(torch.float4_e2m1fn_x2)
asc_s = asc.view(torch.float8_e8m0fnu); bsc_s = bsc.view(torch.float8_e8m0fnu)

def call():
    return aiter.gemm_a4w4(af, bf, asc_s, bsc_s, dtype=torch.bfloat16, bpreshuffle=False)

try:
    out = call(); torch.cuda.synchronize()
except Exception as e:
    print(f"aiter call failed: {e}"); sys.exit(1)

for _ in range(20):
    call()
torch.cuda.synchronize()
ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(100)]
for e0, e1 in ev:
    e0.record(); call(); e1.record()
torch.cuda.synchronize()
ts = sorted(e0.elapsed_time(e1) for e0, e1 in ev)
tf = lambda ms: 2 * M * N * K / (ms * 1e-3) / 1e12
print(f"aiter_a4w4(pre={int(PRE)})  M{M}N{N}K{K}  min {tf(ts[0]):.1f}/med {tf(statistics.median(ts)):.1f} TF")
