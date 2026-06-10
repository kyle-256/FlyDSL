"""Round-26 decisive A/B: kv BN128 gm=4 vs gm=2, INTERLEAVED (cancels thermal
drift), same process, same data. 10 alternating pairs. Win iff gm2 beats gm4 in
the large majority of interleaved pairs (not just mean)."""
import sys, torch
_R = "/workspace/code/FlyDSL"
for p in (_R, _R + "/flydsl/src"):
    sys.path.insert(0, p)
import flydsl.compiler as flyc
from turbo.mxfp4_gemm_8wave import compile_mxfp4_gemm_8w
from turbo.mxfp8_gemm_8wave import preshuffle_scale
SB = 32
N, K, BN = 1024, 8192, 128


def make(M):
    d = "cuda"
    a = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device=d).view(torch.int8).view(-1)
    b = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=d).view(torch.int8).view(-1)
    asc = torch.randint(125, 130, (M, K // SB), dtype=torch.uint8, device=d)
    bsc = torch.randint(125, 130, (N, K // SB), dtype=torch.uint8, device=d)
    asp = preshuffle_scale(asc, K, 4).view(-1)
    bsp = preshuffle_scale(bsc, K, BN // 128).view(-1)
    return a, b, asp, bsp


def t1(cc, ar, it=40):
    e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(); e0.record()
    for _ in range(it):
        cc(*ar)
    e1.record(); torch.cuda.synchronize()
    us = e0.elapsed_time(e1) * 1000 / it
    return 2 * M * N * K / (us / 1e6) / 1e12


for M in (4096, 8192):
    a, b, asp, bsp = make(M)
    st = torch.cuda.current_stream()
    c = torch.zeros((M, N), dtype=torch.bfloat16, device="cuda")
    ar = (a, b, c.view(-1), asp, bsp, M, N, st)
    fn4 = compile_mxfp4_gemm_8w(K=K, BLOCK_M=256, BLOCK_N=BN, mode="pipe", group_m=4, group_n=0)
    fn2 = compile_mxfp4_gemm_8w(K=K, BLOCK_M=256, BLOCK_N=BN, mode="pipe", group_m=2, group_n=0)
    cc4 = flyc.compile(fn4, *ar); cc2 = flyc.compile(fn2, *ar)
    # warmup both
    for _ in range(20):
        cc4(*ar); cc2(*ar)
    torch.cuda.synchronize()
    wins = 0; r4s = []; r2s = []
    for t in range(10):
        r4 = t1(cc4, ar); r2 = t1(cc2, ar)
        r4s.append(r4); r2s.append(r2)
        if r2 > r4:
            wins += 1
    m4 = sum(r4s) / 10; m2 = sum(r2s) / 10
    print(f"kv M={M}: gm4 mean {m4:.0f}  gm2 mean {m2:.0f}  gm2/gm4 {(m2/m4-1)*100:+.1f}%  "
          f"gm2-wins-pairs {wins}/10", flush=True)
    print(f"   gm4 {[f'{x:.0f}' for x in r4s]}", flush=True)
    print(f"   gm2 {[f'{x:.0f}' for x in r2s]}", flush=True)
print("DONE", flush=True)
