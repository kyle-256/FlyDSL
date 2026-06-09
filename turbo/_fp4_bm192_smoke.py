"""Round-30 (B6 r6_2): staged BM192 BN128 narrow-A-G2S correctness.
Compare staged BM192 BN128 (new narrow-A path) vs staged BM256 BN256 reference
(known-good) on kv (N=1024). A-scale n_tiles = BM//64 (BM192->3); pad A-scale
rows to a multiple of 16*n_tiles (BM192 edge-tile). SNR>=40 + det0."""
import sys, torch
_R = "/workspace/code/FlyDSL"
for p in (_R, _R + "/flydsl/src"):
    sys.path.insert(0, p)
import flydsl.compiler as flyc
from turbo.mxfp4_gemm_8wave import compile_mxfp4_gemm_8w
from turbo.mxfp8_gemm_8wave import preshuffle_scale, preshuffle_scale_b_comb
SB = 32


def snr(o, r):
    o = o.float(); r = r.float()
    return 10 * torch.log10((r ** 2).sum() / ((o - r) ** 2).sum().clamp_min(1e-20)).item()


def mkraw(M, N, K):
    d = "cuda"
    a = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device=d)
    b = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=d)
    asc = torch.randint(125, 130, (M, K // SB), dtype=torch.uint8, device=d)
    bsc = torch.randint(125, 130, (N, K // SB), dtype=torch.uint8, device=d)
    return a.view(torch.int8).view(-1), b.view(torch.int8).view(-1), asc, bsc


def ascale(asc, M, K, BM):
    nta = BM // 64
    q = 16 * nta
    pad = ((M + q - 1) // q) * q
    if pad != M:
        ap = torch.zeros((pad, K // SB), dtype=torch.uint8, device=asc.device)
        ap[:M] = asc
        asc = ap
    return preshuffle_scale(asc, K, nta).view(-1)


def build(M, N, K, BM, BN, a, b, asc, bsc):
    asp = ascale(asc, M, K, BM)
    bsp = (preshuffle_scale_b_comb(bsc, K) if BN >= 256 else preshuffle_scale(bsc, K, BN // 128)).view(-1)
    c = torch.zeros((M, N), dtype=torch.bfloat16, device="cuda")
    st = torch.cuda.current_stream()
    fn = compile_mxfp4_gemm_8w(K=K, BLOCK_M=BM, BLOCK_N=BN, mode="staged", group_m=4, group_n=0)
    ar = (a, b, c.view(-1), asp, bsp, M, N, st)
    cc = flyc.compile(fn, *ar)
    cc(*ar); torch.cuda.synchronize()
    return c.clone(), cc, ar


N, K = 1024, 8192
for M in (768, 4096):
    a, b, asc, bsc = mkraw(M, N, K)
    ref, _, _ = build(M, N, K, 256, 256, a, b, asc, bsc)
    out, cc, ar = build(M, N, K, 192, 128, a, b, asc, bsc)
    s = snr(out, ref)
    # det: 2-run quick
    c2 = torch.zeros((M, N), dtype=torch.bfloat16, device="cuda")
    ar2 = (ar[0], ar[1], c2.view(-1), ar[3], ar[4], M, N, ar[7])
    cc(*ar2); torch.cuda.synchronize()
    det2 = torch.equal(out, c2)
    print(f"M={M:5d} BM192/BN128 staged vs BM256/BN256 ref: SNR={s:6.1f} dB  det2={'OK' if det2 else 'NO'}", flush=True)

# strong det0 for the target M=4096 BM192 (300-run fresh data, bitwise)
print("--- det0 300-run fresh kv M4096 BM192/BN128 staged ---", flush=True)
M = 4096
fn = compile_mxfp4_gemm_8w(K=K, BLOCK_M=192, BLOCK_N=128, mode="staged", group_m=4, group_n=0)
maxd = 0; nanflag = False
for run in range(300):
    a, b, asc, bsc = mkraw(M, N, K)
    asp = ascale(asc, M, K, 192)
    bsp = preshuffle_scale(bsc, K, 1).view(-1)
    c1 = torch.zeros((M, N), dtype=torch.bfloat16, device="cuda")
    st = torch.cuda.current_stream()
    ar = (a, b, c1.view(-1), asp, bsp, M, N, st)
    cc = flyc.compile(fn, *ar)
    cc(*ar); torch.cuda.synchronize(); o1 = c1.clone()
    c1.zero_(); cc(*ar); torch.cuda.synchronize()
    maxd = max(maxd, (o1.float() - c1.float()).abs().max().item())
    if torch.isnan(c1).any():
        nanflag = True
print(f"det0: 300-run maxdiff={maxd} nan={nanflag} -> {'DET0' if maxd == 0 and not nanflag else 'FAIL'}", flush=True)
print("DONE", flush=True)
