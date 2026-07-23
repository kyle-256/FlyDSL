"""FlyDSL vs aiter (ck/asm) flash-attn fwd on gfx950, over LLM attention shapes.

Reuses run_config (FlyDSL dualwave) + run_aiter_bench from the repo test so the
FLOP accounting and timing are identical across backends. bf16, causal, batch=1.
Run on the remote gfx950 node (chi2774 / mlperf_gptoss); see FLYDSL_VS_AITER_ATTN.md.
"""
import sys, json
sys.path.insert(0, "/workspace/code/FlyDSL")
import torch
from tests.kernels.test_flash_attn_fwd import run_config, run_aiter_bench

BF16 = torch.bfloat16
# (name, h_q, h_kv, d)  -- FlyDSL dense needs symmetric d; asymmetric/512 handled as errors.
MODELS = [
    ("Qwen3-235B", 64, 4, 128),
    ("Qwen3-32B", 64, 8, 128),
    ("GPT-OSS", 64, 8, 64),
    ("DeepSeek-V4-Flash", 64, 1, 512),
    ("DeepSeek-V4-Pro", 128, 1, 512),
]
SEQS = [2048, 4096, 8192]


def safe(fn):
    try:
        return fn()
    except Exception as e:
        return {"err": f"{type(e).__name__}: {str(e)[:120]}"}


def tf(r):
    return r.get("tflops")


rows = []
print(f"{'model':18}{'seq':>6}{'fly_TF':>9}{'ck_TF':>9}{'asm_TF':>9}{'fly_ok':>8}")
for name, h, hkv, d in MODELS:
    for s in SEQS:
        fly = safe(lambda: run_config(1, s, h, d, BF16, True, warmup=10, iters=30,
                                      dtype_str="bf16", verbose=False, num_kv_heads=hkv))
        ck = safe(lambda: run_aiter_bench(1, s, h, d, BF16, True, 10, 30, backend="ck", num_kv_heads=hkv))
        asm = safe(lambda: run_aiter_bench(1, s, h, d, BF16, True, 10, 30, backend="asm", num_kv_heads=hkv))
        print(f"{name:18}{s:>6}"
              f"{(tf(fly) or 0):>9.0f}{(tf(ck) or 0):>9.0f}{(tf(asm) or 0):>9.0f}"
              f"{str(fly.get('passed')):>8}", flush=True)
        rows.append({"model": name, "s": s, "h": h, "hkv": hkv, "d": d,
                     "fly_tf": tf(fly), "ck_tf": tf(ck), "asm_tf": tf(asm),
                     "fly_ok": fly.get("passed"), "fly_err": fly.get("err"),
                     "ck_err": ck.get("err"), "asm_err": asm.get("err")})
print("ROWS_JSON " + json.dumps(rows, default=str))
