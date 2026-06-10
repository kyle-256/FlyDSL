"""Step 5 of the bare-asm mxfp4 campaign: MULTIPLE distinct accumulators in one
asm hw-loop, with ds_read_b128 issued per-iter and the MFMAs run BACK-TO-BACK with
NO s_nop (distinct accs are mutually independent, so the result->read RAW hazard
that forced s_nop in the single-acc case is gone; an acc is only re-touched after
NACC-1 other MFMAs + the ds_reads + addr advance, which fully covers MFMA latency).

This is the >pipe lever: the gaps that single-acc had to fill with s_nop become the
slots to issue ds_read. Here we first prove the multi-acc + ds_read loop is
bit-exact (NA x NB outer product), then later interleave reads into MFMA gaps.

LDS (Int32, one wave/64 lanes): A region = NA tiles, each N K-blocks, each 256 i32
(=1024B); tile i, block k, lane L -> i32 index (i*N+k)*256 + L*4. Per-tile addr reg
advances +1024B per iter. B region symmetric (NB tiles).
"""
import sys, torch
_R = "/workspace/code/FlyDSL"
for p in (_R, _R + "/flydsl/src"): sys.path.insert(0, p)
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import arith, buffer_ops, gpu, rocdl, range_constexpr, const_expr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from kernels.fp8_gemm_utils import pack_i32x4_i32x8


def _raw(v):
    if not isinstance(v, ir.Value) and hasattr(v, "ir_value"): return v.ir_value()
    return v


N = 8          # K-blocks accumulated
NA = 8         # A tiles
NB = 4         # B tiles
NACC = NA * NB
BLK = 1024     # bytes per K-block in LDS
MFMA = "v_mfma_scale_f32_16x16x128_f8f6f4 ${d}, ${a}, ${b}, ${d}, ${sa}, ${sb} op_sel_hi:[0,0,0] cbsz:4 blgp:4"


def _layout():
    """Pure-python compile-time operand/$-index layout + asm/cons/struct strings.
    Outputs (struct order): acc_q(NACC vec4f32), afrag_i(NA vec4i32), bfrag_j(NB
    vec4i32), cnt(i32), addr_a_i(NA i32), addr_b_j(NB i32). Inputs (operand order):
    a_init(NA), b_init(NB), sa, sb, nval, stride, zeros(NACC)."""
    o_acc = list(range(NACC))
    o_af = list(range(NACC, NACC + NA))
    o_bf = list(range(NACC + NA, NACC + NA + NB))
    o_cnt = NACC + NA + NB
    o_aa = [o_cnt + 1 + i for i in range(NA)]
    o_ba = [o_cnt + 1 + NA + j for j in range(NB)]
    n_out = NACC + NA + NB + 1 + NA + NB
    i_sa = n_out + NA + NB; i_sb = i_sa + 1; i_nval = i_sa + 2; i_stride = i_sa + 3

    out_types = ["vector<4xf32>"] * NACC + ["vector<4xi32>"] * (NA + NB) + ["i32"] + ["i32"] * (NA + NB)
    st_str = "!llvm.struct<(" + ", ".join(out_types) + ")>"
    out_cons = ["=v"] * NACC + ["=&v"] * (NA + NB) + ["=&s"] + ["=v"] * (NA + NB)
    in_cons = ([str(x) for x in o_aa] + [str(x) for x in o_ba] + ["v", "v", "s", "s"] + [str(q) for q in o_acc])
    cons = ",".join(out_cons + in_cons)

    L = [f"s_mov_b32 ${o_cnt}, 0", "1:"]
    for i in range(NA): L.append(f"ds_read_b128 ${o_af[i]}, ${o_aa[i]}")
    for j in range(NB): L.append(f"ds_read_b128 ${o_bf[j]}, ${o_ba[j]}")
    L.append("s_waitcnt lgkmcnt(0)")
    for i in range(NA):                       # distinct accs back-to-back, NO s_nop
        for j in range(NB):
            L.append(MFMA.format(d=o_acc[i * NB + j], a=o_af[i], b=o_bf[j], sa=i_sa, sb=i_sb))
    for i in range(NA): L.append(f"v_add_u32 ${o_aa[i]}, ${o_aa[i]}, ${i_stride}")
    for j in range(NB): L.append(f"v_add_u32 ${o_ba[j]}, ${o_ba[j]}, ${i_stride}")
    L.append(f"s_add_u32 ${o_cnt}, ${o_cnt}, 1")
    L.append(f"s_cmp_lt_u32 ${o_cnt}, ${i_nval}")
    L.append("s_cbranch_scc1 1b")
    return "\n".join(L), cons, st_str


def build(mode):
    asm, cons, st_str = _layout()
    NTI = NA * N * 256; NTB = NB * N * 256

    @fx.struct
    class Storage:
        A_lds: fx.Array[fx.Int32, NTI, 16]
        B_lds: fx.Array[fx.Int32, NTB, 16]

    @flyc.kernel(name=f"asm5_{mode}", known_block_size=[64, 1, 1])
    def k(out: fx.Tensor, av: fx.Tensor, bv: fx.Tensor, sav: fx.Int32, sbv: fx.Int32, nval: fx.Int32, stride: fx.Int32):
        lane = arith.index_cast(T.i32, gpu.thread_id("x"))
        res4 = Vec.make_type(4, fx.Float32)
        zero = Vec.filled(4, 0.0, fx.Float32)
        z4 = Vec.filled(4, 0, fx.Int32)

        def gload(t, blk):
            bi = buffer_ops.extract_base_index(t, address_space=1)
            base = _llvm.inttoptr(ir.Type.parse("!llvm.ptr<1>"), _raw(fx.Int64(bi)))
            boff = (blk * fx.Int32(256) + lane * fx.Int32(4)) * fx.Int32(4)
            return _llvm.load(T.vec(4, T.i32), buffer_ops.get_element_ptr(base, byte_offset=_raw(boff), elem_type=T.i8))

        if const_expr(mode == "ref"):
            accs = [zero] * NACC
            for kk in range_constexpr(N):
                A = [gload(av, i * N + kk) for i in range_constexpr(NA)]
                B = [gload(bv, j * N + kk) for j in range_constexpr(NB)]
                for i in range_constexpr(NA):
                    for j in range_constexpr(NB):
                        q = i * NB + j
                        accs[q] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                            res4, [pack_i32x4_i32x8(Vec(A[i]), z4), pack_i32x4_i32x8(Vec(B[j]), z4), accs[q], 4, 4, 0, sav, 0, sbv])
            outc = [Vec(a) for a in accs]
        else:
            lds = fx.SharedAllocator().allocate(Storage).peek()

            def lds_store(field, idx_i32, vec_raw):
                pp = fx.add_offset(field.ptr, fx.make_int_tuple(idx_i32))
                fx.make_view(pp, fx.make_layout(4, 1)).store(Vec(vec_raw))
            for kk in range_constexpr(N):
                for i in range_constexpr(NA):
                    lds_store(lds.A_lds, (i * N + kk) * 256 + lane * fx.Int32(4), gload(av, i * N + kk))
                for j in range_constexpr(NB):
                    lds_store(lds.B_lds, (j * N + kk) * 256 + lane * fx.Int32(4), gload(bv, j * N + kk))
            _llvm.inline_asm(ir.Type.parse("!llvm.void"), [], "s_waitcnt lgkmcnt(0)\ns_barrier", "", has_side_effects=True)

            def addr(field, tile):
                pp = fx.add_offset(field.ptr, fx.make_int_tuple(tile * N * 256 + lane * fx.Int32(4)))
                return _raw(fx.ptrtoint(pp))
            a_init = [addr(lds.A_lds, i) for i in range_constexpr(NA)]
            b_init = [addr(lds.B_lds, j) for j in range_constexpr(NB)]

            st = ir.Type.parse(st_str)
            ops = ([_raw(x) for x in a_init] + [_raw(x) for x in b_init]
                   + [_raw(sav), _raw(sbv), _raw(nval), _raw(stride)] + [_raw(zero)] * NACC)
            r = _llvm.inline_asm(st, ops, asm, cons, has_side_effects=True)
            outc = [Vec(_llvm.extractvalue(res4, r, [q])) for q in range_constexpr(NACC)]

        bo = buffer_ops.extract_base_index(out, address_space=1)
        base = _llvm.inttoptr(ir.Type.parse("!llvm.ptr<1>"), _raw(fx.Int64(bo)))
        for q in range_constexpr(NACC):
            for i in range_constexpr(4):
                off = ((fx.Int32(q) * fx.Int32(64) + lane) * fx.Int32(4) + fx.Int32(i)) * fx.Int32(4)
                _llvm.StoreOp(_raw(outc[q][i]), buffer_ops.get_element_ptr(base, byte_offset=_raw(off), elem_type=T.i8))

    @flyc.jit
    def launch(out: fx.Tensor, av: fx.Tensor, bv: fx.Tensor, sav: fx.Int32, sbv: fx.Int32, nval: fx.Int32, stride: fx.Int32, stream: fx.Stream):
        k(out, av, bv, sav, sbv, nval, stride).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)
    return launch


def main():
    d = "cuda"; torch.manual_seed(0)
    av = torch.randint(0, 2**31, (NA * N * 256,), dtype=torch.int32, device=d)
    bv = torch.randint(0, 2**31, (NB * N * 256,), dtype=torch.int32, device=d)
    sa = 0x7f7f7f7f; stm = torch.cuda.current_stream(); res = {}
    for mode in ("ref", "dsasm"):
        out = torch.zeros(NACC * 64 * 4, dtype=torch.float32, device=d)
        cc = flyc.compile(build(mode), out, av, bv, sa, sa, N, BLK, stm)
        out.zero_(); cc(out, av, bv, sa, sa, N, BLK, stm); torch.cuda.synchronize(); res[mode] = out.clone()
    diff = (res["ref"] - res["dsasm"]).abs().max().item()
    print(f"N={N} NA={NA} NB={NB} NACC={NACC} max|ref-dsasm|={diff:.3g} match={diff<1e-2} ref[:4]={res['ref'][:4].tolist()}")


main()
