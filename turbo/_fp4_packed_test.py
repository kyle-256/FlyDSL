"""Packed-scale (opsel) vs broadcast-scale pipe: correctness (vs f32 ref) + perf.

packed_scale packs n_tiles E8M0 into ONE i32 (byte t = tile t) and selects via
opsel_a/opsel_b per XDL -> 1 dword/(region,k) instead of broadcast dwordx4 (4x
less scale VMEM). This is the official CK approach. Target: recover the 8-17%
scale-path cost measured by the const_scale diagnostic.
"""
import sys, math, statistics
import torch
_R = "/workspace/code/FlyDSL"
for p in (_R, _R + "/flydsl/src"):
    sys.path.insert(0, p)
import flydsl.compiler as flyc
from tests.kernels.utils import fp4_utils
from turbo.mxfp4_gemm_8wave import compile_mxfp4_gemm_8w, recommend_config, preshuffle_mxfp4_scales

SB = 32


def ref_mxfp4(a_u8, b_u8, asc, bsc, M, N, K):
    a_f = fp4_utils.mxfp4_to_f32(a_u8)[:M, :K].float()
    b_f = fp4_utils.mxfp4_to_f32(b_u8)[:N, :K].float()
    a_s = fp4_utils.e8m0_to_f32(asc).repeat_interleave(SB, -1)[:M, :K].float()
    b_s = fp4_utils.e8m0_to_f32(bsc).repeat_interleave(SB, -1)[:N, :K].float()
    return torch.matmul(a_f * a_s, (b_f * b_s).T)


def snr(o, r):
    o, r = o.float(), r.float()
    n = (o - r).pow(2).mean().item(); s = r.pow(2).mean().item()
    return 99.0 if n == 0 else 10 * math.log10(s / max(n, 1e-30))


def meas(cc, ar, it=40):
    e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(); e0.record()
    for _ in range(it): cc(*ar)
    e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1) * 1000 / it


def args(c, a, b, sa, sb, M, N):
    return (a.view(torch.int8).view(-1), b.view(torch.int8).view(-1), c.view(-1),
            sa.view(-1), sb.view(-1), M, N, torch.cuda.current_stream())


SHAPES = [
    ("7Bqo",   4096, 4096, 4096),
    ("7Bdn",   4096, 4096, 11008),
    ("70Bqo",  4096, 8192, 8192),
    ("70Bdn",  4096, 8192, 28672),
]
ROUNDS = 7

for tag, M, N, K in SHAPES:
    BM, BN, gm, gn, nx = recommend_config(M, N, K)
    if BN < 256: BN = 256; gm = 4
    d = "cuda"
    a = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device=d)
    b = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=d)
    asc = torch.randint(125, 130, (M, K // SB), dtype=torch.uint8, device=d)
    bsc = torch.randint(125, 130, (N, K // SB), dtype=torch.uint8, device=d)
    ref = ref_mxfp4(a, b, asc, bsc, M, N, K)

    sa_bc, sb_bc = preshuffle_mxfp4_scales(asc, bsc, K, BLOCK_M=BM, BLOCK_N=BN, mode="staged")  # broadcast
    sa_pk, sb_pk = preshuffle_mxfp4_scales(asc, bsc, K, BLOCK_M=BM, BLOCK_N=BN, mode="pipe", combine_a=False)  # packed 3-load
    sa_ca, sb_ca = preshuffle_mxfp4_scales(asc, bsc, K, BLOCK_M=BM, BLOCK_N=BN, mode="pipe")     # combine-A (default) 2-load
    variants = {
        "bcast":  (dict(mode="pipe", packed_scale=False), sa_bc, sb_bc),
        "packed": (dict(mode="pipe", combine_a=False), sa_pk, sb_pk),
        "combA":  (dict(mode="pipe"), sa_ca, sb_ca),
    }
    res = {}
    for vn, (kw, sa, sb) in variants.items():
        fn = compile_mxfp4_gemm_8w(K=K, BLOCK_M=BM, BLOCK_N=BN, group_m=gm, group_n=gn, num_xcds=nx, **kw)
        c = torch.zeros((M, N), dtype=torch.bfloat16, device=d)
        ar = args(c, a, b, sa, sb, M, N)
        cc = flyc.compile(fn, *ar); c.zero_(); cc(*ar); torch.cuda.synchronize()
        res[vn] = (cc, ar, c.clone(), snr(c, ref))

    for vn in res:
        cc, ar, _, _ = res[vn]
        for _ in range(20): cc(*ar)
    torch.cuda.synchronize()
    samp = {vn: [] for vn in res}
    for r in range(ROUNDS):
        for vn in res:
            samp[vn].append(meas(res[vn][0], res[vn][1]))
    base = statistics.median(samp["bcast"])
    print(f"=== {tag} M{M} N{N} K{K} BM{BM} BN{BN} ===")
    for vn in res:
        med = statistics.median(samp[vn]); tf = 2 * M * N * K / (med / 1e6) / 1e12
        print(f"  {vn:7s} med={med:8.2f}us tf={tf:6.0f} speedup={base/med:.4f} SNR={res[vn][3]:.1f}")
    sys.stdout.flush()
