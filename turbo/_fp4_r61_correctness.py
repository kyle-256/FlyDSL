"""round-61 correctness gate: the big-K band route (gn=nb//8 for K>=11008) is a
pure tile->CU permutation, so it must be (a) bit-exact vs the production gn0 path
(itself det0-sealed at r39) and (b) det0 under fresh-data multi-run. Check both
newly-routed shapes (7B down, 70B down) at both M."""
import sys, torch
_R = "/workspace/code/FlyDSL"
for p in (_R, _R + "/flydsl/src"):
    sys.path.insert(0, p)
import flydsl.compiler as flyc
from turbo.mxfp4_gemm_8wave import compile_mxfp4_gemm_8w
from turbo.mxfp8_gemm_8wave import preshuffle_scale, preshuffle_scale_b_comb
SB = 32


def mkraw(M, N, K):
    d = "cuda"
    a = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device=d)
    b = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=d)
    asc = torch.randint(120, 135, (M, K // SB), dtype=torch.uint8, device=d)
    bsc = torch.randint(120, 135, (N, K // SB), dtype=torch.uint8, device=d)
    return a.view(torch.int8).view(-1), b.view(torch.int8).view(-1), asc, bsc


def ascale(asc, M, K):
    return preshuffle_scale(asc, K, 4).view(-1)  # BM=256 -> nta=4


def build(M, N, K, a, b, asc, bsc, gn):
    asp = ascale(asc, M, K)
    bsp = preshuffle_scale_b_comb(bsc, K).view(-1)
    c = torch.zeros((M, N), dtype=torch.bfloat16, device="cuda")
    st = torch.cuda.current_stream()
    fn = compile_mxfp4_gemm_8w(K=K, BLOCK_M=256, BLOCK_N=256, mode="pipe", group_m=4, group_n=gn)
    ar = (a, b, c.view(-1), asp, bsp, M, N, st)
    cc = flyc.compile(fn, *ar); cc(*ar); torch.cuda.synchronize()
    return cc, ar, c


ok = True
for tag, N, K in [("70Bdown", 8192, 28672), ("7Bdown", 4096, 11008)]:
    gn = (N // 256) // 8
    for M in (4096, 8192):
        a, b, asc, bsc = mkraw(M, N, K)
        _, _, c0 = build(M, N, K, a, b, asc, bsc, 0)
        ccb, arb, cb = build(M, N, K, a, b, asc, bsc, gn)
        bitdiff = (cb.float() - c0.float()).abs().max().item()
        # det0: 300-run fresh-data on the band config
        md = 0.0
        for r in range(300):
            a2, b2, asc2, bsc2 = mkraw(M, N, K)
            asp = ascale(asc2, M, K); bsp = preshuffle_scale_b_comb(bsc2, K).view(-1)
            cref = torch.zeros((M, N), dtype=torch.bfloat16, device="cuda")
            ar2 = (a2, b2, cref.view(-1), asp, bsp, M, N, torch.cuda.current_stream())
            ccb(*ar2); torch.cuda.synchronize(); first = cref.clone()
            ccb(*ar2); torch.cuda.synchronize()
            md = max(md, (cref.float() - first.float()).abs().max().item())
            if md > 0:
                break
        status = "PASS" if (bitdiff == 0 and md == 0) else "FAIL"
        ok = ok and (bitdiff == 0 and md == 0)
        print(f"{tag} M={M} gn={gn}: band-vs-gn0 bitdiff={bitdiff:.4g}  det0(300run fresh) maxdiff={md:.4g}  {status}", flush=True)
print("ALL_PASS" if ok else "SOME_FAIL", flush=True)
