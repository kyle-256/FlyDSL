"""Round-26 det0 gate: kv BN128 group_m=2, 300-run fresh-data bitwise, both M."""
import sys, torch
_R = "/workspace/code/FlyDSL"
for p in (_R, _R + "/flydsl/src"):
    sys.path.insert(0, p)
import flydsl.compiler as flyc
from turbo.mxfp4_gemm_8wave import compile_mxfp4_gemm_8w
from turbo.mxfp8_gemm_8wave import preshuffle_scale
SB = 32
N, K, BN, GM = 1024, 8192, 128, 2
fn = compile_mxfp4_gemm_8w(K=K, BLOCK_M=256, BLOCK_N=BN, mode="pipe", group_m=GM, group_n=0)
for M in (4096, 8192):
    maxd = 0; nanflag = False
    for run in range(300):
        d = "cuda"
        a = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device=d).view(torch.int8).view(-1)
        b = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=d).view(torch.int8).view(-1)
        asc = torch.randint(125, 130, (M, K // SB), dtype=torch.uint8, device=d)
        bsc = torch.randint(125, 130, (N, K // SB), dtype=torch.uint8, device=d)
        asp = preshuffle_scale(asc, K, 4).view(-1)
        bsp = preshuffle_scale(bsc, K, BN // 128).view(-1)
        c1 = torch.zeros((M, N), dtype=torch.bfloat16, device=d)
        st = torch.cuda.current_stream()
        ar = (a, b, c1.view(-1), asp, bsp, M, N, st)
        cc = flyc.compile(fn, *ar)
        cc(*ar); torch.cuda.synchronize(); o1 = c1.clone()
        c1.zero_(); cc(*ar); torch.cuda.synchronize()
        d_ = (o1.float() - c1.float()).abs().max().item()
        maxd = max(maxd, d_)
        if torch.isnan(c1).any(): nanflag = True
    print(f"kv M={M} BN={BN} gm={GM}: 300-run maxdiff={maxd} nan={nanflag} -> {'DET0' if maxd==0 and not nanflag else 'FAIL'}", flush=True)
print("DONE", flush=True)
