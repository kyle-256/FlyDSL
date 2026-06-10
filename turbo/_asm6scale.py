"""Step 6a: real per-lane E8M0 scales fed to the asm MFMA, varying per K-iter.

Step-5 proved the i32 scale operand path with a constant 0x7f7f7f7f. The only
unverified pieces for integration are (1) genuine non-trivial per-lane scale values
and (2) scale advancing per K-iter. Here each (tile, k, lane) gets a distinct E8M0
byte (broadcast to i32, matching ScaleS2R's broadcast_u8_to_u32 + opsel0 byte-0
sampling). Scales are staged in LDS and ds_read_b32'd inside the asm loop with their
own advancing address (stride 256B = 64 lanes x 4B), alongside the b128 A/B frags.
Compared bit-exact vs an intrinsic reference using the same scales. acc -> AGPR.
"""
import sys, os, torch
ACC_AGPR = os.environ.get("ACC_AGPR", "1") == "1"
_R = "/workspace/code/FlyDSL"
for p in (_R, _R + "/flydsl/src"): sys.path.insert(0, p)
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import arith, buffer_ops, gpu, rocdl, range_constexpr, const_expr, primitive
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from kernels.fp8_gemm_utils import pack_i32x4_i32x8


def _raw(v):
    if not isinstance(v, ir.Value) and hasattr(v, "ir_value"): return v.ir_value()
    return v


N = 8
NA = 2
NB = 2
NACC = NA * NB
AB_STRIDE = 1024   # bytes per K-block (A/B frag)
SC_STRIDE = 256    # bytes per K-iter scale (64 lanes x 4B)
MFMA = "v_mfma_scale_f32_16x16x128_f8f6f4 ${d}, ${a}, ${b}, ${d}, ${sa}, ${sb} op_sel_hi:[0,0,0] cbsz:4 blgp:4"


def _layout():
    nab = NA + NB
    o_acc = list(range(NACC)); base = NACC
    o_af = [base + i for i in range(NA)]; base += NA
    o_bf = [base + j for j in range(NB)]; base += NB
    o_saf = [base + i for i in range(NA)]; base += NA   # scaleA frag (i32 in vgpr)
    o_sbf = [base + j for j in range(NB)]; base += NB
    o_cnt = base; base += 1
    o_aa = [base + i for i in range(NA)]; base += NA     # A/B frag addrs
    o_ba = [base + j for j in range(NB)]; base += NB
    o_saa = [base + i for i in range(NA)]; base += NA    # scale addrs
    o_sba = [base + j for j in range(NB)]; base += NB
    n_out = base
    # inputs: a_init(NA),b_init(NB),sa_init(NA),sb_init(NB),nval,stride_ab,stride_sc,zeros(NACC)
    i_nval = n_out + 2 * nab
    i_sab = i_nval + 1; i_ssc = i_nval + 2

    out_types = (["vector<4xf32>"] * NACC + ["vector<4xi32>"] * nab + ["i32"] * nab + ["i32"]
                 + ["i32"] * nab + ["i32"] * nab)
    st_str = "!llvm.struct<(" + ", ".join(out_types) + ")>"
    acc_c = "=a" if ACC_AGPR else "=v"
    out_cons = ([acc_c] * NACC + ["=&v"] * nab + ["=&v"] * nab + ["=&s"] + ["=v"] * nab + ["=v"] * nab)
    in_cons = ([str(x) for x in o_aa] + [str(x) for x in o_ba] + [str(x) for x in o_saa] + [str(x) for x in o_sba]
               + ["s", "s", "s"] + [str(q) for q in o_acc])
    cons = ",".join(out_cons + in_cons)

    L = [f"s_mov_b32 ${o_cnt}, 0", "1:"]
    for i in range(NA): L.append(f"ds_read_b128 ${o_af[i]}, ${o_aa[i]}")
    for j in range(NB): L.append(f"ds_read_b128 ${o_bf[j]}, ${o_ba[j]}")
    for i in range(NA): L.append(f"ds_read_b32 ${o_saf[i]}, ${o_saa[i]}")
    for j in range(NB): L.append(f"ds_read_b32 ${o_sbf[j]}, ${o_sba[j]}")
    L.append("s_waitcnt lgkmcnt(0)")
    for i in range(NA):
        for j in range(NB):
            L.append(MFMA.format(d=o_acc[i * NB + j], a=o_af[i], b=o_bf[j], sa=o_saf[i], sb=o_sbf[j]))
    for i in range(NA): L.append(f"v_add_u32 ${o_aa[i]}, ${o_aa[i]}, ${i_sab}")
    for j in range(NB): L.append(f"v_add_u32 ${o_ba[j]}, ${o_ba[j]}, ${i_sab}")
    for i in range(NA): L.append(f"v_add_u32 ${o_saa[i]}, ${o_saa[i]}, ${i_ssc}")
    for j in range(NB): L.append(f"v_add_u32 ${o_sba[j]}, ${o_sba[j]}, ${i_ssc}")
    L.append(f"s_add_u32 ${o_cnt}, ${o_cnt}, 1")
    L.append(f"s_cmp_lt_u32 ${o_cnt}, ${i_nval}")
    L.append("s_cbranch_scc1 1b")
    return "\n".join(L), cons, st_str


def build(mode):
    asm, cons, st_str = _layout()
    NTI = NA * N * 256; NTB = NB * N * 256
    SAI = NA * N * 64; SBI = NB * N * 64

    @fx.struct
    class Storage:
        A_lds: fx.Array[fx.Int32, NTI, 16]
        B_lds: fx.Array[fx.Int32, NTB, 16]
        SA_lds: fx.Array[fx.Int32, SAI, 16]
        SB_lds: fx.Array[fx.Int32, SBI, 16]

    @flyc.kernel(name=f"asm6_{mode}", known_block_size=[64, 1, 1])
    def k(out: fx.Tensor, av: fx.Tensor, bv: fx.Tensor, sa: fx.Tensor, sb: fx.Tensor, nval: fx.Int32, st_ab: fx.Int32, st_sc: fx.Int32):
        lane = arith.index_cast(T.i32, gpu.thread_id("x"))
        res4 = Vec.make_type(4, fx.Float32)
        zero = Vec.filled(4, 0.0, fx.Float32)
        z4 = Vec.filled(4, 0, fx.Int32)

        def gload4(t, blk):  # i32x4 A/B frag
            bi = buffer_ops.extract_base_index(t, address_space=1)
            base = _llvm.inttoptr(ir.Type.parse("!llvm.ptr<1>"), _raw(fx.Int64(bi)))
            boff = (blk * fx.Int32(256) + lane * fx.Int32(4)) * fx.Int32(4)
            return _llvm.load(T.vec(4, T.i32), buffer_ops.get_element_ptr(base, byte_offset=_raw(boff), elem_type=T.i8))

        def gload1(t, blk):  # single i32 scale at (blk, lane)
            bi = buffer_ops.extract_base_index(t, address_space=1)
            base = _llvm.inttoptr(ir.Type.parse("!llvm.ptr<1>"), _raw(fx.Int64(bi)))
            boff = (blk * fx.Int32(64) + lane) * fx.Int32(4)
            return _llvm.load(T.i32, buffer_ops.get_element_ptr(base, byte_offset=_raw(boff), elem_type=T.i8))

        if const_expr(mode == "ref"):
            accs = [zero] * NACC
            for kk in range_constexpr(N):
                A = [gload4(av, i * N + kk) for i in range_constexpr(NA)]
                B = [gload4(bv, j * N + kk) for j in range_constexpr(NB)]
                SA = [gload1(sa, i * N + kk) for i in range_constexpr(NA)]
                SB = [gload1(sb, j * N + kk) for j in range_constexpr(NB)]
                for i in range_constexpr(NA):
                    for j in range_constexpr(NB):
                        q = i * NB + j
                        accs[q] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                            res4, [pack_i32x4_i32x8(Vec(A[i]), z4), pack_i32x4_i32x8(Vec(B[j]), z4), accs[q], 4, 4, 0, SA[i], 0, SB[j]])
            outc = [Vec(a) for a in accs]
        else:
            lds = fx.SharedAllocator().allocate(Storage).peek()

            def st4(field, idx, vraw):
                pp = fx.add_offset(field.ptr, fx.make_int_tuple(idx))
                fx.make_view(pp, fx.make_layout(4, 1)).store(Vec(vraw))

            def st1(field, idx, vraw):
                pp = fx.add_offset(field.ptr, fx.make_int_tuple(idx))
                primitive.ptr_store(vraw, pp)
            for kk in range_constexpr(N):
                for i in range_constexpr(NA):
                    st4(lds.A_lds, (i * N + kk) * 256 + lane * fx.Int32(4), gload4(av, i * N + kk))
                    st1(lds.SA_lds, (i * N + kk) * 64 + lane, gload1(sa, i * N + kk))
                for j in range_constexpr(NB):
                    st4(lds.B_lds, (j * N + kk) * 256 + lane * fx.Int32(4), gload4(bv, j * N + kk))
                    st1(lds.SB_lds, (j * N + kk) * 64 + lane, gload1(sb, j * N + kk))
            _llvm.inline_asm(ir.Type.parse("!llvm.void"), [], "s_waitcnt lgkmcnt(0)\ns_barrier", "", has_side_effects=True)

            def adr(field, eoff):
                pp = fx.add_offset(field.ptr, fx.make_int_tuple(eoff))
                return _raw(fx.ptrtoint(pp))
            a_in = [adr(lds.A_lds, i * N * 256 + lane * fx.Int32(4)) for i in range_constexpr(NA)]
            b_in = [adr(lds.B_lds, j * N * 256 + lane * fx.Int32(4)) for j in range_constexpr(NB)]
            sa_in = [adr(lds.SA_lds, i * N * 64 + lane) for i in range_constexpr(NA)]
            sb_in = [adr(lds.SB_lds, j * N * 64 + lane) for j in range_constexpr(NB)]

            st = ir.Type.parse(st_str)
            ops = ([_raw(x) for x in a_in] + [_raw(x) for x in b_in] + [_raw(x) for x in sa_in] + [_raw(x) for x in sb_in]
                   + [_raw(nval), _raw(st_ab), _raw(st_sc)] + [_raw(zero)] * NACC)
            r = _llvm.inline_asm(st, ops, asm, cons, has_side_effects=True)
            outc = [Vec(_llvm.extractvalue(res4, r, [q])) for q in range_constexpr(NACC)]

        bo = buffer_ops.extract_base_index(out, address_space=1)
        base = _llvm.inttoptr(ir.Type.parse("!llvm.ptr<1>"), _raw(fx.Int64(bo)))
        for q in range_constexpr(NACC):
            for i in range_constexpr(4):
                off = ((fx.Int32(q) * fx.Int32(64) + lane) * fx.Int32(4) + fx.Int32(i)) * fx.Int32(4)
                _llvm.StoreOp(_raw(outc[q][i]), buffer_ops.get_element_ptr(base, byte_offset=_raw(off), elem_type=T.i8))

    @flyc.jit
    def launch(out: fx.Tensor, av: fx.Tensor, bv: fx.Tensor, sa: fx.Tensor, sb: fx.Tensor, nval: fx.Int32, st_ab: fx.Int32, st_sc: fx.Int32, stream: fx.Stream):
        k(out, av, bv, sa, sb, nval, st_ab, st_sc).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)
    return launch


def main():
    d = "cuda"; torch.manual_seed(0)
    av = torch.randint(0, 2**31, (NA * N * 256,), dtype=torch.int32, device=d)
    bv = torch.randint(0, 2**31, (NB * N * 256,), dtype=torch.int32, device=d)
    # E8M0 byte in [120,134] (~2^-7..2^7), broadcast to i32, per (tile,k,lane)
    eb_a = torch.randint(120, 135, (NA * N * 64,), dtype=torch.int32, device=d)
    eb_b = torch.randint(120, 135, (NB * N * 64,), dtype=torch.int32, device=d)
    bc = 0x01010101
    sa = (eb_a * bc).to(torch.int32); sb = (eb_b * bc).to(torch.int32)
    stm = torch.cuda.current_stream(); res = {}
    for mode in ("ref", "asm"):
        out = torch.zeros(NACC * 64 * 4, dtype=torch.float32, device=d)
        cc = flyc.compile(build(mode), out, av, bv, sa, sb, N, AB_STRIDE, SC_STRIDE, stm)
        out.zero_(); cc(out, av, bv, sa, sb, N, AB_STRIDE, SC_STRIDE, stm); torch.cuda.synchronize(); res[mode] = out.clone()
    diff = (res["ref"] - res["asm"]).abs().max().item()
    rel = diff / (res["ref"].abs().max().item() + 1e-9)
    print(f"N={N} NA={NA} NB={NB} max|ref-asm|={diff:.3g} rel={rel:.3g} match={rel<1e-5} ref[:4]={res['ref'][:4].tolist()}")


main()
