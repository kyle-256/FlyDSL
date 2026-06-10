"""Step 4 of the bare-asm mxfp4 campaign: move ds_read_b128 INSIDE the inline-asm
hw-loop. LDS holds N K-blocks; the asm loop reads block k via ds_read_b128 with a
per-iter advancing LDS address (+1024B), feeds the fp4 scale-MFMA, accumulates the
SAME acc across iters. Compared bit-exact vs an intrinsic reference that loads the
same N blocks from global and accumulates with N intrinsic MFMAs.

LDS layout (one wave, 64 lanes): block k, lane L -> byte (k*1024 + L*16), 16B each.
Global av/bv: block k, lane L -> i32x4 at byte (k*64+L)*16.
"""
import sys, torch
_R = "/workspace/code/FlyDSL"
for p in (_R, _R + "/flydsl/src"): sys.path.insert(0, p)
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import arith, buffer_ops, gpu, rocdl, range_constexpr, primitive, const_expr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from kernels.fp8_gemm_utils import pack_i32x4_i32x8


def _raw(v):
    if not isinstance(v, ir.Value) and hasattr(v, "ir_value"): return v.ir_value()
    return v


N = 8                # number of K-blocks accumulated
BLK_BYTES = 64 * 16  # 1024B per K-block in LDS (64 lanes x 16B)


def build(mode):
    @fx.struct
    class SharedStorageDs:                       # i32 elems: N*256 per buf (=N*1024B)
        A_lds: fx.Array[fx.Int32, N * 256, 16]
        B_lds: fx.Array[fx.Int32, N * 256, 16]

    @flyc.kernel(name=f"asmds_{mode}", known_block_size=[64, 1, 1])
    def k(out: fx.Tensor, av: fx.Tensor, bv: fx.Tensor, sav: fx.Int32, sbv: fx.Int32, nval: fx.Int32, stride: fx.Int32):
        lane = arith.index_cast(T.i32, gpu.thread_id("x"))
        res4 = Vec.make_type(4, fx.Float32)
        zero = Vec.filled(4, 0.0, fx.Float32)
        z4 = Vec.filled(4, 0, fx.Int32)

        def gload(t, blk):  # i32x4 from global block `blk` at this lane
            bi = buffer_ops.extract_base_index(t, address_space=1)
            base = _llvm.inttoptr(ir.Type.parse("!llvm.ptr<1>"), _raw(fx.Int64(bi)))
            boff = (blk * fx.Int32(64) + lane) * fx.Int32(16)
            return _llvm.load(T.vec(4, T.i32),
                              buffer_ops.get_element_ptr(base, byte_offset=_raw(boff), elem_type=T.i8))

        if const_expr(mode == "ref"):
            c = zero
            for blk in range_constexpr(N):
                a = gload(av, blk); b = gload(bv, blk)
                c = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                    res4, [pack_i32x4_i32x8(Vec(a), z4), pack_i32x4_i32x8(Vec(b), z4), c, 4, 4, 0, sav, 0, sbv])
            outc = Vec(c)
        else:
            lds = fx.SharedAllocator().allocate(SharedStorageDs).peek()
            # write N blocks into LDS (this lane's 16B per block)
            def lds_store(field, blk, vec_raw):  # offset in i32 elements
                pp = fx.add_offset(field.ptr, fx.make_int_tuple(blk * 256 + lane * fx.Int32(4)))
                fx.make_view(pp, fx.make_layout(4, 1)).store(Vec(vec_raw))
            for blk in range_constexpr(N):
                lds_store(lds.A_lds, blk, gload(av, blk))
                lds_store(lds.B_lds, blk, gload(bv, blk))
            # barrier: LDS writes visible before asm ds_read
            _llvm.inline_asm(ir.Type.parse("!llvm.void"), [], "s_waitcnt lgkmcnt(0)\ns_barrier", "", has_side_effects=True)

            def lds_addr(field):  # i32 LDS byte offset of this lane's block-0 slot
                pp = fx.add_offset(field.ptr, fx.make_int_tuple(lane * fx.Int32(4)))
                a = fx.ptrtoint(pp)
                a = _raw(a)
                if str(a.type) != "i32":
                    a = arith.index_cast(T.i32, a) if str(a.type) == "index" else _llvm.trunc(T.i32, a)
                return a
            a0 = lds_addr(lds.A_lds); b0 = lds_addr(lds.B_lds)

            st = ir.Type.parse("!llvm.struct<(vector<4xf32>, vector<4xi32>, vector<4xi32>, i32, i32, i32)>")
            asm = ("s_mov_b32 $3, 0\n"
                   "1:\n"
                   "ds_read_b128 $1, $4\n"
                   "ds_read_b128 $2, $5\n"
                   "s_waitcnt lgkmcnt(0)\n"
                   "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, $0, $8, $9 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n"
                   "s_nop 8\n"
                   "v_add_u32 $4, $4, $11\n"
                   "v_add_u32 $5, $5, $11\n"
                   "s_add_u32 $3, $3, 1\n"
                   "s_cmp_lt_u32 $3, $10\n"
                   "s_cbranch_scc1 1b")
            cons = "=v,=&v,=&v,=&s,=v,=v,4,5,v,v,s,s,0"
            r = _llvm.inline_asm(st, [_raw(a0), _raw(b0), _raw(sav), _raw(sbv), _raw(nval), _raw(stride), _raw(zero)],
                                 asm, cons, has_side_effects=True)
            outc = Vec(_llvm.extractvalue(res4, r, [0]))

        bo = buffer_ops.extract_base_index(out, address_space=1)
        base = _llvm.inttoptr(ir.Type.parse("!llvm.ptr<1>"), _raw(fx.Int64(bo)))
        for i in range_constexpr(4):
            _llvm.StoreOp(_raw(outc[i]),
                          buffer_ops.get_element_ptr(base, byte_offset=_raw((lane * fx.Int32(4) + fx.Int32(i)) * fx.Int32(4)), elem_type=T.i8))

    @flyc.jit
    def launch(out: fx.Tensor, av: fx.Tensor, bv: fx.Tensor, sav: fx.Int32, sbv: fx.Int32, nval: fx.Int32, stride: fx.Int32, stream: fx.Stream):
        k(out, av, bv, sav, sbv, nval, stride).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)
    return launch


def main():
    d = "cuda"; torch.manual_seed(0)
    av = torch.randint(0, 2**31, (N * 64 * 4,), dtype=torch.int32, device=d)
    bv = torch.randint(0, 2**31, (N * 64 * 4,), dtype=torch.int32, device=d)
    sa = 0x7f7f7f7f; stm = torch.cuda.current_stream(); res = {}
    for mode in ("ref", "dsasm"):
        out = torch.zeros(64 * 4, dtype=torch.float32, device=d)
        cc = flyc.compile(build(mode), out, av, bv, sa, sa, N, BLK_BYTES, stm)
        out.zero_(); cc(out, av, bv, sa, sa, N, BLK_BYTES, stm); torch.cuda.synchronize(); res[mode] = out.clone()
    diff = (res["ref"] - res["dsasm"]).abs().max().item()
    print(f"N={N} max|ref-dsasm|={diff:.3g} match={diff<1e-2} ref[:4]={res['ref'][:4].tolist()} dsasm[:4]={res['dsasm'][:4].tolist()}")


main()
