"""Round-46 B8 r8_3: pipe3 minimal-barrier (1 sync/iter) — correctness (SNR + det0
300-run FRESH) + perf A/B vs pipe. Bulk shapes. GPU7 only.
det0 here = the CRITICAL race gate (3-stage barrier removal): for each of 3 FRESH
inputs, run pipe3 100x, require all bitwise identical (race is data-dependent)."""
import os, sys, math, statistics, torch
_R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for p in (_R, os.path.join(_R, "flydsl", "src")):
    if p not in sys.path:
        sys.path.insert(0, p)
import flydsl.compiler as flyc
from flydsl.runtime.device import get_rocm_arch
from tests.kernels.utils import fp4_utils
from turbo.mxfp4_gemm_8wave import compile_mxfp4_gemm_8w, recommend_config
from turbo.mxfp8_gemm_8wave import preshuffle_scale, preshuffle_scale_b_comb
SB = 32


def ascale(asc, M, K, BM):
    nta = BM // 64; q = 16 * nta
    pad = ((M + q - 1) // q) * q
    if pad != M:
        ap = torch.zeros((pad, K // SB), dtype=torch.uint8, device=asc.device); ap[:M] = asc; asc = ap
    return preshuffle_scale(asc, K, nta).view(-1)


def bscale(bsc, K, BN):
    return (preshuffle_scale_b_comb(bsc, K) if BN >= 256 else preshuffle_scale(bsc, K, BN // 128)).view(-1)


def snr_db(out, ref):
    out, ref = out.float(), ref.float()
    n = (out - ref).pow(2).mean(); s = ref.pow(2).mean()
    return float("inf") if n.item() == 0 else (10 * torch.log10(s / n)).item()


def mk(M, N, K, dev="cuda"):
    a = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device=dev)
    b = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=dev)
    asc = torch.randint(124, 131, (M, K // SB), dtype=torch.uint8, device=dev)
    bsc = torch.randint(124, 131, (N, K // SB), dtype=torch.uint8, device=dev)
    return a, b, asc, bsc


def build(M, N, K, BM, BN, gm, gn, mode):
    fn = compile_mxfp4_gemm_8w(K=K, BLOCK_M=BM, BLOCK_N=BN, mode=mode, group_m=gm, group_n=gn)
    def args(a, b, c, sa, sb):
        return (a.view(torch.int8).view(-1), b.view(torch.int8).view(-1), c.view(-1),
                sa, sb, M, N, torch.cuda.current_stream())
    return fn, args


def time_once(fn, it=50):
    e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(); e0.record()
    for _ in range(it): fn()
    e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1) * 1000 / it


SHAPES = [("7B q/o", 4096, 4096, 4096), ("70B q/o", 8192, 8192, 8192), ("70B down", 8192, 8192, 28672)]


def main():
    arch = str(get_rocm_arch()); assert "gfx95" in arch, f"needs gfx950, got {arch}"
    dev = "cuda"
    print(f"{'shape':12s}{'M':>6}{'N':>6}{'K':>6}  {'SNR3':>6s}  {'det0(3x100 fresh)':>18s}  {'pipe_us':>8s}{'pipe3_us':>9s}  {'ratio':>6s}")
    lr = 0.0; n = 0; nwin = 0; allok = True
    for tag, M, N, K in SHAPES:
        BM, BN, gm, gn = recommend_config(M, N, K)
        a, b, asc, bsc = mk(M, N, K, dev)
        ref = torch.matmul(
            fp4_utils.mxfp4_to_f32(a)[:M, :K].float() * fp4_utils.e8m0_to_f32(asc).repeat_interleave(SB, -1)[:M, :K].float(),
            (fp4_utils.mxfp4_to_f32(b)[:N, :K].float() * fp4_utils.e8m0_to_f32(bsc).repeat_interleave(SB, -1)[:N, :K].float()).T)
        sa = ascale(asc, M, K, BM); sb = bscale(bsc, K, BN)
        try:
            c3 = torch.zeros((M, N), dtype=torch.bfloat16, device=dev)
            fn3, mk3 = build(M, N, K, BM, BN, gm, gn, "pipe3"); ar3 = mk3(a, b, c3, sa, sb)
            cc3 = flyc.compile(fn3, *ar3); cc3(*ar3); torch.cuda.synchronize()
            s3 = snr_db(c3, ref)
            cpa = torch.zeros((M, N), dtype=torch.bfloat16, device=dev)
            fnp, mkp = build(M, N, K, BM, BN, gm, gn, "pipe"); arp = mkp(a, b, cpa, sa, sb)
            ccp = flyc.compile(fnp, *arp); ccp(*arp); torch.cuda.synchronize()
        except Exception as e:
            print(f"{tag:12s}{M:>6}{N:>6}{K:>6}  ERR {type(e).__name__}: {e}"); allok = False; continue
        # det0: 3 FRESH inputs x 100 reruns bitwise (race is data-dependent)
        maxd = 0.0
        for _ in range(3):
            af, bf, ascf, bscf = mk(M, N, K, dev)
            saf = ascale(ascf, M, K, BM); sbf = bscale(bscf, K, BN)
            cf = torch.zeros((M, N), dtype=torch.bfloat16, device=dev)
            arf = mk3(af, bf, cf, saf, sbf)
            cc3(*arf); torch.cuda.synchronize(); c_first = cf.clone()
            for _ in range(100):
                cc3(*arf); torch.cuda.synchronize()
                d = (cf.float() - c_first.float()).abs().max().item()
                if d > maxd: maxd = d
                if maxd > 0: break
            if maxd > 0: break
        det = "DET0" if maxd == 0 else f"NONDET({maxd:.3g})"
        ok = (s3 >= 40 and maxd == 0); allok = allok and ok
        # perf A/B interleaved median-of-7
        fp = lambda: ccp(*arp); fh = lambda: cc3(*ar3)
        for _ in range(10): fp(); fh()
        torch.cuda.synchronize()
        rr = []; pus = []; h3 = []
        for _ in range(7):
            tp = time_once(fp); th = time_once(fh)
            rr.append(tp / th); pus.append(tp); h3.append(th)
        r = statistics.median(rr); lr += math.log(r); n += 1; nwin += (r > 1.01)
        print(f"{tag:12s}{M:>6}{N:>6}{K:>6}  {s3:>6.1f}  {det:>18s}  {statistics.median(pus)*1e3:>8.1f}{statistics.median(h3)*1e3:>9.1f}  {r:>6.3f}")
    if n:
        print(f"\nGEOMEAN pipe3/pipe = {math.exp(lr/n):.4f} over {n} bulk; pipe3 wins(>1.01): {nwin}/{n}")
    print("CORRECTNESS:", "PASS" if allok else "FAIL (SNR<40 or NONDET)")


if __name__ == "__main__":
    main()
