"""FlyDSL flash-attention (fwd only) across the same LLM shapes as the aiter bench.

FlyDSL's flydsl_flash_attn_func is forward-only (gfx950 DUALWAVE_SWP dense), so this
mirrors the aiter benchmark's *forward* path for an apples-to-apples fwd TFLOPS
comparison. Dense mode assumes d_qk == d_v (single head_dim, mult of 32, >=64), so
DeepSeek-V3 (asymmetric qk192/v128) and DeepSeek-V4 (hd512) are attempted but
expected to be unsupported.

Prints a table and a trailing JSON blob (rows) for downstream .md aggregation.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # repo root for `import kernels`
import torch

from kernels.flash_attn_interface import flydsl_flash_attn_func

BF16 = torch.bfloat16

# (name, h_q, h_kv, d_qk, d_v, causal, note) -- identical to the aiter bench.
MODELS = [
    ("Qwen3-235B-A22B", 64, 4, 128, 128, True, "GQA 64/4"),
    ("Qwen3-32B", 64, 8, 128, 128, True, "GQA 64/8"),
    ("DeepSeek-V3 (MLA)", 128, 128, 192, 128, True, "MHA 128, qk192/v128 (asymmetric)"),
    ("DeepSeek-V4-Flash", 64, 1, 512, 512, True, "MQA 64/1, hd512 (hybrid sparse; dense proxy)"),
    ("DeepSeek-V4-Pro", 128, 1, 512, 512, True, "MQA 128/1, hd512 (hybrid sparse; dense proxy)"),
    ("GPT-OSS-20b/120b", 64, 8, 64, 64, True, "GQA 64/8, +SWA/sink (full-attn layer)"),
]


def robust_time(fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    e0 = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    e1 = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        e0[i].record()
        fn()
        e1[i].record()
    torch.cuda.synchronize()
    ts = sorted(e0[i].elapsed_time(e1[i]) for i in range(iters))  # ms
    return ts[len(ts) // 2]


def bench_one(b, s, h_q, h_kv, d_qk, d_v, causal):
    if d_qk != d_v:
        raise NotImplementedError(f"FlyDSL dense flash-attn requires d_qk==d_v (got {d_qk}/{d_v})")
    d = d_qk
    torch.manual_seed(0)
    q = torch.randn(b, s, h_q, d, device="cuda", dtype=BF16)
    k = torch.randn(b, s, h_kv, d, device="cuda", dtype=BF16)
    v = torch.randn(b, s, h_kv, d, device="cuda", dtype=BF16)

    def fwd():
        return flydsl_flash_attn_func(q, k, v, causal=causal, num_kv_heads=h_kv)

    fwd()  # build + warm the cache
    fwd_ms = robust_time(fwd)
    fwd_flops = 2.0 * b * h_q * s * s * (d_qk + d_v) * (0.5 if causal else 1.0)
    return {"fwd_ms": fwd_ms, "fwd_tflops": fwd_flops / (fwd_ms * 1e-3) / 1e12}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqlens", default="2048,4096,8192")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    seqlens = [int(x) for x in args.seqlens.split(",")]
    sel = set(args.only.split(",")) if args.only else None

    rows = []
    print(f"{'model':24} {'seq':>6} {'fwd_us':>9} {'fwd_TF':>8}")
    for name, h_q, h_kv, d_qk, d_v, causal, note in MODELS:
        if sel is not None and name not in sel:
            continue
        for s in seqlens:
            try:
                r = bench_one(args.batch, s, h_q, h_kv, d_qk, d_v, causal)
                print(f"{name:24} {s:>6} {r['fwd_ms']*1e3:>9.1f} {r['fwd_tflops']:>8.0f}")
                rows.append({"model": name, "note": note, "b": args.batch, "s": s,
                             "h_q": h_q, "h_kv": h_kv, "d_qk": d_qk, "d_v": d_v, "causal": causal,
                             "ok": True, **{k: round(vv, 3) for k, vv in r.items()}})
            except Exception as e:
                if os.environ.get("BENCH_TB"):
                    import traceback
                    traceback.print_exc()
                print(f"{name:24} {s:>6}  FAILED: {type(e).__name__}: {str(e)[:100]}")
                rows.append({"model": name, "note": note, "b": args.batch, "s": s,
                             "h_q": h_q, "h_kv": h_kv, "d_qk": d_qk, "d_v": d_v, "causal": causal,
                             "ok": False, "err": f"{type(e).__name__}: {str(e)[:200]}"})
                torch.cuda.empty_cache()
    print("ROWS_JSON " + json.dumps(rows))


if __name__ == "__main__":
    main()
