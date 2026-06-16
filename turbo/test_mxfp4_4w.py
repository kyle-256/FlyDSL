"""4-wave mxfp4 correctness + perf + AGPR/spill check.
Usage: python turbo/test_mxfp4_4w.py [M N K] [agpr0]"""
import sys, torch, statistics
_R = "/workspace/code/FlyDSL"
for p in (_R, _R + "/flydsl/src"):
    sys.path.insert(0, p)
import flydsl.compiler as flyc
from turbo.mxfp4_gemm_4wave import compile_mxfp4_gemm_4w, preshuffle_mxfp4_scales_4w

SB = 32
M, N, K = (int(x) for x in (sys.argv[1:4] or [8192, 8192, 28672]))
_flags = set(sys.argv[4:])
agpr = "agpr0" not in _flags
pad = "pad0" not in _flags
import os
WAIT = int(os.environ.get("WAIT", "0"))
MAXNREG = int(os.environ.get("MAXNREG", "0"))
BLOCK_K = int(os.environ.get("BLOCK_K", "256"))
BLOCK_N = int(os.environ.get("BLOCK_N4", "256"))
GM4 = int(os.environ.get("GM4", "4")); GN4 = int(os.environ.get("GN4", "16")); NX4 = int(os.environ.get("NX4", "8"))
d = "cuda"; torch.manual_seed(0)
a = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device=d)
b = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=d)
asc = torch.randint(125, 130, (M, K // SB), dtype=torch.uint8, device=d)
if os.environ.get("FP4_CONSTSC","0")=="1":
    asc = torch.full((M, K // SB), 127, dtype=torch.uint8, device=d)
bsc = torch.randint(125, 130, (N, K // SB), dtype=torch.uint8, device=d)
if os.environ.get("FP4_CONSTSC","0")=="1":
    bsc = torch.full((N, K // SB), 127, dtype=torch.uint8, device=d)
ai = a.view(torch.int8).view(-1); bi = b.view(torch.int8).view(-1); st = torch.cuda.current_stream()

asp, bsp = preshuffle_mxfp4_scales_4w(asc, bsc, K)
c = torch.zeros((M, N), dtype=torch.bfloat16, device=d)
fn = compile_mxfp4_gemm_4w(K=K, agpr=agpr, pad=pad, wait=WAIT, maxnreg=MAXNREG, block_k=BLOCK_K, BLOCK_N=BLOCK_N, group_m=GM4, group_n=GN4, num_xcds=NX4)
ar = (ai, bi, c.view(-1), asp.view(-1), bsp.view(-1), M, N, st)
cc = flyc.compile(fn, *ar)
cc(*ar); torch.cuda.synchronize(); c0 = c.clone()
det = 0; detmax = 0.0
for _ in range(int(os.environ.get("DETRUNS", "1"))):
    c.zero_(); cc(*ar); torch.cuda.synchronize(); det += (c0 != c).sum().item()
    detmax = max(detmax, (c0.float() - c.float()).abs().max().item())
if os.environ.get("FP4_LOC"): print(f"DETMAX(max |c0-c| across reruns) = {detmax:.4f}  (c0 mean-abs = {c0.float().abs().mean().item():.2f})")

def lut_f32(x):
    x = x.repeat_interleave(2, dim=1).clone()
    x[:, ::2] = x[:, ::2] & 0xF; x[:, 1::2] = x[:, 1::2] >> 4
    lut = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0, -.5, -1, -1.5, -2, -3, -4, -6], dtype=torch.float32, device=x.device)
    return lut[x.long()]
REF = (lut_f32(a) * 2.0 ** (asc.repeat_interleave(SB, 1).float() - 127)) @ (lut_f32(b) * 2.0 ** (bsc.repeat_interleave(SB, 1).float() - 127)).T
snr = 10 * torch.log10((REF**2).mean() / ((c0.float() - REF)**2).mean()).item()
if os.environ.get("FP4_LOC"):
    err = (c0.float() - REF).abs()
    rb = err.view(M//16, 16, N).mean(dim=(1,2)); cb = err.view(M, N//16, 16).mean(dim=(0,2))
    print("LOC: worst row-blocks(16):", [int(i) for i in rb.topk(8).indices.tolist()], "vals", [f"{v:.2f}" for v in rb.topk(8).values.tolist()])
    print("LOC: worst col-blocks(16):", [int(i) for i in cb.topk(8).indices.tolist()], "vals", [f"{v:.2f}" for v in cb.topk(8).values.tolist()])
    print("LOC: rowblock err frac>0.1:", int((rb>0.1).sum()), "/", M//16, " colblock:", int((cb>0.1).sum()), "/", N//16)
    # per-128-col band (N tile = wave_n coverage) and per-256 (BLOCK_N)
    cb256 = err.view(M, N//256, 256).mean(dim=(0,2)); print("LOC: err per 256-N-block (first 8):", [f"{v:.2f}" for v in cb256[:8].tolist()])
for _ in range(20):
    cc(*ar)
torch.cuda.synchronize()
ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(100)]
for e0, e1 in ev:
    e0.record(); cc(*ar); e1.record()
torch.cuda.synchronize()
ts = sorted(e0.elapsed_time(e1) for e0, e1 in ev)
tf = lambda ms: 2 * M * N * K / (ms * 1e-3) / 1e12
print(f"4w(agpr{int(agpr)}pad{int(pad)}w{WAIT})  M{M}N{N}K{K}  SNR {snr:.1f}dB  det {det}  "
      f"min {tf(ts[0]):.1f}/med {tf(statistics.median(ts)):.1f} TF")
