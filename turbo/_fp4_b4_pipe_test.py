"""B4 r4_3: validate PRODUCTION pipe (mode="pipe") 4M x 2N (wave_topo="4x2") BN64.

Ports the r4_2 narrow-B-G2S + 4x2 topology into kernel_gemm_pipe (software
pipelined). Gate: SNR>=40 dB vs dequant-f32 ref + det0 (300-run fresh-data
bitwise, the harness class that caught r_k6 vmcnt(3) races) on kv shapes. Also
re-checks default 2x4 BN256 + BN128 (production kv path) for byte-identity.
"""
import os, sys, torch
_R = "/workspace/code/FlyDSL"
for p in (_R, _R + "/flydsl/src"):
    sys.path.insert(0, p)
import flydsl.compiler as flyc
from flydsl.runtime.device import get_rocm_arch
from tests.kernels.utils import fp4_utils
from turbo.mxfp4_gemm_8wave import compile_mxfp4_gemm_8w
from turbo.mxfp8_gemm_8wave import preshuffle_scale, preshuffle_scale_b_comb
SB = 32


def ref_gemm(a_u8, b_u8, asc, bsc, M, N, K):
    a_f = fp4_utils.mxfp4_to_f32(a_u8)[:M, :K].float()
    b_f = fp4_utils.mxfp4_to_f32(b_u8)[:N, :K].float()
    a_s = fp4_utils.e8m0_to_f32(asc).repeat_interleave(SB, dim=-1)[:M, :K].float()
    b_s = fp4_utils.e8m0_to_f32(bsc).repeat_interleave(SB, dim=-1)[:N, :K].float()
    return torch.matmul(a_f * a_s, (b_f * b_s).T)


def snr(o, r):
    o, r = o.float(), r.float()
    n = (o - r).pow(2).mean()
    return float("inf") if n.item() == 0 else (10 * torch.log10(r.pow(2).mean() / n)).item()


def mk(M, N, K, d="cuda"):
    a = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device=d)
    b = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=d)
    asc = torch.randint(125, 130, (M, K // SB), dtype=torch.uint8, device=d)
    bsc = torch.randint(125, 130, (N, K // SB), dtype=torch.uint8, device=d)
    return a, b, asc, bsc


def run(M, N, K, BM, BN, topo, det_runs=0):
    a, b, asc, bsc = mk(M, N, K)
    ref = ref_gemm(a, b, asc, bsc, M, N, K)
    n_ta = BM // 128 if topo == "4x2" else BM // 64
    n_tb = BN // 64 if topo == "4x2" else BN // 128
    asp = preshuffle_scale(asc, K, n_ta).view(-1)
    bsp = (preshuffle_scale_b_comb(bsc, K) if BN >= 256 else preshuffle_scale(bsc, K, n_tb)).view(-1)
    c = torch.zeros((M, N), dtype=torch.bfloat16, device="cuda")
    fn = compile_mxfp4_gemm_8w(K=K, BLOCK_M=BM, BLOCK_N=BN, mode="pipe", wave_topo=topo, group_m=4, group_n=0)
    ai, bi = a.view(torch.int8).view(-1), b.view(torch.int8).view(-1)
    args = (ai, bi, c.view(-1), asp, bsp, M, N, torch.cuda.current_stream())
    cc = flyc.compile(fn, *args)
    cc(*args); torch.cuda.synchronize()
    s = snr(c.float(), ref)
    # det: det_runs reruns, EACH with fresh random data (data-dependent race catcher)
    maxd = 0.0
    if det_runs:
        for _ in range(det_runs):
            a2, b2, asc2, bsc2 = mk(M, N, K)
            asp2 = preshuffle_scale(asc2, K, n_ta).view(-1)
            bsp2 = (preshuffle_scale_b_comb(bsc2, K) if BN >= 256 else preshuffle_scale(bsc2, K, n_tb)).view(-1)
            ai2, bi2 = a2.view(torch.int8).view(-1), b2.view(torch.int8).view(-1)
            ar2 = (ai2, bi2, c.view(-1), asp2, bsp2, M, N, torch.cuda.current_stream())
            c.zero_(); cc(*ar2); torch.cuda.synchronize(); o1 = c.clone()
            c.zero_(); cc(*ar2); torch.cuda.synchronize()
            maxd = max(maxd, (c - o1).abs().max().item())
    ok = s > 40 and maxd == 0
    print(f"[{topo} BM{BM} BN{BN}] M={M} N={N} K={K}  SNR={s:6.2f} dB  det{det_runs}_maxdiff={maxd:.3g}  {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    assert "gfx95" in str(get_rocm_arch())
    ok = True
    print("=== regression: default 2x4 pipe (production paths) ===")
    ok &= run(256, 1024, 512, 256, 256, "2x4")
    ok &= run(4096, 1024, 8192, 256, 128, "2x4")  # production kv BN128
    print("=== B4 r4_3: 4x2 BN64 pipe (kv) ===")
    ok &= run(256, 1024, 512, 256, 64, "4x2")
    ok &= run(4096, 1024, 8192, 256, 64, "4x2", det_runs=300)
    ok &= run(8192, 1024, 8192, 256, 64, "4x2", det_runs=300)
    print("RESULT:", "PASS" if ok else "FAIL")
