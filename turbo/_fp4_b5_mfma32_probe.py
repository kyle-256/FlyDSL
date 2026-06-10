"""B5 r_5a foundational gate: does v_mfma_scale_f32_32x32x64_f8f6f4 assemble via
FlyDSL inline-asm? (No rocdl binding exists — only 16x16x128.) 32x32x64 fp4:
A/B = i32x4 per lane (32 fp4, same as 16x16x128), result = v16f32 (32x32/64
lanes = 16 f32/lane, vs 16x16's v4f32). If it compiles + launches without
INVALID_ISA, the building block is viable and B5 can proceed to layout work."""
import sys, torch
_R = "/workspace/code/FlyDSL"
for p in (_R, _R + "/flydsl/src"):
    sys.path.insert(0, p)
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import buffer_ops
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import _to_raw

_ASM = "v_mfma_scale_f32_32x32x64_f8f6f4 $0, $1, $2, 0, $3, $4 op_sel_hi:[0,0,0] cbsz:4 blgp:4"
_CONS = "=a,v,v,v,v"


@flyc.kernel
def probe32(A: fx.Tensor, B: fx.Tensor, SA: fx.Tensor, SB: fx.Tensor, OUT: fx.Tensor):
    rA = buffer_ops.create_buffer_resource(A)
    rB = buffer_ops.create_buffer_resource(B)
    rSA = buffer_ops.create_buffer_resource(SA)
    rSB = buffer_ops.create_buffer_resource(SB)
    rOUT = buffer_ops.create_buffer_resource(OUT)
    lane = fx.thread_idx.x % 64
    a = Vec(buffer_ops.buffer_load(rA, lane * 4, vec_width=4, dtype=fx.Int32))
    b = Vec(buffer_ops.buffer_load(rB, lane * 4, vec_width=4, dtype=fx.Int32))
    sa = buffer_ops.buffer_load(rSA, lane, vec_width=1, dtype=fx.Int32)
    sb = buffer_ops.buffer_load(rSB, lane, vec_width=1, dtype=fx.Int32)
    res_ty = Vec.make_type(16, fx.Float32)
    d = _llvm.inline_asm(res_ty, [_to_raw(a), _to_raw(b), _to_raw(sa), _to_raw(sb)], _ASM, _CONS, has_side_effects=False)
    dv = Vec(d)
    for i in range(16):
        buffer_ops.buffer_store(dv[i], rOUT, lane * 16 + i)


@flyc.jit
def launch(A: fx.Tensor, B: fx.Tensor, SA: fx.Tensor, SB: fx.Tensor, OUT: fx.Tensor,
           stream: fx.Stream = fx.Stream(None)):
    probe32(A, B, SA, SB, OUT).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)


if __name__ == "__main__":
    d = "cuda"
    A = torch.randint(0, 127, (64 * 4,), dtype=torch.int32, device=d)
    B = torch.randint(0, 127, (64 * 4,), dtype=torch.int32, device=d)
    SA = torch.full((64,), 127, dtype=torch.int32, device=d)
    SB = torch.full((64,), 127, dtype=torch.int32, device=d)
    OUT = torch.zeros((64 * 16,), dtype=torch.float32, device=d)
    st = torch.cuda.current_stream()
    cc = flyc.compile(launch, A, B, SA, SB, OUT, st)
    cc(A, B, SA, SB, OUT, st)
    torch.cuda.synchronize()
    print("ASSEMBLE+LAUNCH OK — v_mfma_scale_f32_32x32x64_f8f6f4 is viable via FlyDSL inline-asm")
    print("OUT[:8] =", OUT[:8].tolist())
