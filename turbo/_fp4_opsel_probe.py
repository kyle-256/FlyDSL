"""Definitively map mfma_scale opsel -> which byte of the i32 scale operand.

One 16x16x128 f8f6f4 MFMA. A,B = fp4 1.0 (code 0x22 packs two 1.0 nibbles, E2M1
1.0 = 0b010 -> nibble 0x2). Unscaled dot over K=128 = 128. scale_a packed i32 =
bytes [127,128,129,130] (E8M0 exps 0,1,2,3 -> 2^0,2^1,2^2,2^3). scale_b = 127
(1.0) broadcast. All lanes get the SAME packed scale_a, so opsel=OP makes every
sub-block use byte OP => output[0,0] = 128 * 2^OP. Run OP=0..3 -> read mapping.
"""
import sys
import torch
_R = "/workspace/code/FlyDSL"
for p in (_R, _R + "/flydsl/src"):
    sys.path.insert(0, p)
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import buffer_ops, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from kernels.fp8_gemm_utils import pack_i32x4_i32x8


def build(OP):
    @flyc.kernel(name=f"opsel_{OP}", known_block_size=[64, 1, 1])
    def k(out: fx.Tensor, av: fx.Tensor, bv: fx.Tensor, sa: fx.Tensor, sb: fx.Tensor):
        lane = fx.thread_idx.x % 64
        ra = buffer_ops.create_buffer_resource(av, max_size=True)
        rb = buffer_ops.create_buffer_resource(bv, max_size=True)
        rsa = buffer_ops.create_buffer_resource(sa, max_size=True)
        rsb = buffer_ops.create_buffer_resource(sb, max_size=True)
        z4 = Vec.filled(4, 0, fx.Int32)
        a4 = Vec(buffer_ops.buffer_load(ra, lane * fx.Int32(4), vec_width=4, dtype=T.i32))
        b4 = Vec(buffer_ops.buffer_load(rb, lane * fx.Int32(4), vec_width=4, dtype=T.i32))
        a8 = pack_i32x4_i32x8(a4, z4); b8 = pack_i32x4_i32x8(b4, z4)
        sav = buffer_ops.buffer_load(rsa, lane, vec_width=1, dtype=T.i32)
        sbv = buffer_ops.buffer_load(rsb, lane, vec_width=1, dtype=T.i32)
        c = Vec.filled(4, 0.0, fx.Float32)
        r = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
            Vec.make_type(4, fx.Float32), [a8, b8, c, 4, 4, OP, sav, 0, sbv])
        rv = Vec(r)
        bo = buffer_ops.create_buffer_resource(out, max_size=True)
        from flydsl.expr import range_constexpr
        for i in range_constexpr(4):
            buffer_ops.buffer_store(rv[i], bo, lane * fx.Int32(4) + fx.Int32(i))

    @flyc.jit
    def launch(out, av, bv, sa, sb, stream):
        k(out, av, bv, sa, sb).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)
    return launch


d = "cuda"
# fp4 1.0 = nibble 0x2; byte 0x22 = two 1.0; 128 fp4 = 64 bytes = 16 i32 per row.
av = torch.full((64 * 4,), 0x22222222, dtype=torch.int32, device=d)
bv = torch.full((64 * 4,), 0x22222222, dtype=torch.int32, device=d)
# scale_a packed bytes [127,128,129,130]; scale_b = 0x7F broadcast (1.0)
sa_bytes = torch.tensor([127, 128, 129, 130], dtype=torch.uint8, device=d).repeat(64)
sa = sa_bytes.view(64, 4).contiguous().view(torch.int32).view(-1)
sb = torch.full((64,), 0x7F7F7F7F, dtype=torch.int64, device=d).to(torch.int32)
stm = torch.cuda.current_stream()
for OP in range(4):
    out = torch.zeros(64 * 4, dtype=torch.float32, device=d)
    cc = flyc.compile(build(OP), out, av, bv, sa, sb, stm)
    out.zero_(); cc(out, av, bv, sa, sb, stm); torch.cuda.synchronize()
    v = out[0].item()
    exp = None
    for e in range(4):
        if abs(v - 128 * (2 ** e)) < 1.0:
            exp = e
    print(f"opsel={OP}: out[0]={v:.1f}  => reads byte {exp} (2^{exp})" if exp is not None else f"opsel={OP}: out[0]={v:.1f} (unmatched)")
