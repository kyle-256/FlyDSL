"""Phase 6b-0: cross-iter AGPR threading for Option B (per-iter asm cluster).

Option B keeps the K-loop in Python (range_constexpr) and emits ONE asm block per
iter: all ds_reads + scale reads + the 32 distinct MFMAs (no s_nop) + a single
lgkmcnt(0). The 32 accumulators must stay in AGPR ACROSS iter boundaries (between
separate asm blocks) -- the one risk the whole-loop POC didn't cover (R13-adjacent).
Iter 0 uses C==0 literal (fresh =a, no tie, per MfmaScaleFp4 note); iters>0 tie acc
in/out so LLVM keeps them resident in AGPR. Validated bit-exact + spill via ISA.
"""
import sys, os, torch
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


N = 4           # K-iters (Python-unrolled)
NA = 8
NB = 4
NACC = NA * NB  # 32
MFMA = "v_mfma_scale_f32_16x16x128_f8f6f4 ${d}, ${a}, ${b}, ${c}, ${sa}, ${sb} op_sel_hi:[0,0,0] cbsz:4 blgp:4"


def _layout():
    """One iter's asm (acc ALWAYS tied -> iter0 ties zero, iters>0 tie prev acc;
    matches the _asm5il =a form that worked). Outputs: acc(NACC =a), af,bf,saf,sbf
    temps. Inputs: aa,ba,saa,sba addrs + acc_in(NACC tie to acc)."""
    nab = NA + NB
    b = 0
    o_acc = list(range(b, b + NACC)); b += NACC
    o_af = list(range(b, b + NA)); b += NA
    o_bf = list(range(b, b + NB)); b += NB
    o_saf = list(range(b, b + NA)); b += NA
    o_sbf = list(range(b, b + NB)); b += NB
    n_out = b
    i_aa = n_out; i_ba = i_aa + NA; i_saa = i_ba + NB; i_sba = i_saa + NA

    out_types = ["vector<4xf32>"] * NACC + ["vector<4xi32>"] * nab + ["i32"] * nab
    st_str = "!llvm.struct<(" + ", ".join(out_types) + ")>"
    out_cons = ["=a"] * NACC + ["=&v"] * nab + ["=&v"] * nab
    in_cons = ["v"] * (2 * nab) + [str(q) for q in o_acc]
    cons = ",".join(out_cons + in_cons)

    L = []
    for i in range(NA): L.append(f"ds_read_b128 ${o_af[i]}, ${i_aa + i}")
    for j in range(NB): L.append(f"ds_read_b128 ${o_bf[j]}, ${i_ba + j}")
    for i in range(NA): L.append(f"ds_read_b32 ${o_saf[i]}, ${i_saa + i}")
    for j in range(NB): L.append(f"ds_read_b32 ${o_sbf[j]}, ${i_sba + j}")
    L.append("s_waitcnt lgkmcnt(0)")
    for i in range(NA):
        for j in range(NB):
            q = i * NB + j
            L.append(MFMA.format(d=o_acc[q], a=o_af[i], b=o_bf[j], c=o_acc[q], sa=o_saf[i], sb=o_sbf[j]))
    return "\n".join(L), cons, st_str


def build(mode):
    asmS, consS, ststrS = _layout()
    NTI = NA * N * 256; NTB = NB * N * 256
    SAI = NA * N * 64; SBI = NB * N * 64

    @fx.struct
    class Storage:
        A_lds: fx.Array[fx.Int32, NTI, 16]
        B_lds: fx.Array[fx.Int32, NTB, 16]
        SA_lds: fx.Array[fx.Int32, SAI, 16]
        SB_lds: fx.Array[fx.Int32, SBI, 16]

    @flyc.kernel(name=f"asm6b_{mode}", known_block_size=[64, 1, 1])
    def k(out: fx.Tensor, av: fx.Tensor, bv: fx.Tensor, sa: fx.Tensor, sb: fx.Tensor):
        lane = arith.index_cast(T.i32, gpu.thread_id("x"))
        res4 = Vec.make_type(4, fx.Float32)
        zero = Vec.filled(4, 0.0, fx.Float32)
        z4 = Vec.filled(4, 0, fx.Int32)

        def gload4(t, blk):
            bi = buffer_ops.extract_base_index(t, address_space=1)
            base = _llvm.inttoptr(ir.Type.parse("!llvm.ptr<1>"), _raw(fx.Int64(bi)))
            boff = (blk * fx.Int32(256) + lane * fx.Int32(4)) * fx.Int32(4)
            return _llvm.load(T.vec(4, T.i32), buffer_ops.get_element_ptr(base, byte_offset=_raw(boff), elem_type=T.i8))

        def gload1(t, blk):
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
                fx.make_view(fx.add_offset(field.ptr, fx.make_int_tuple(idx)), fx.make_layout(4, 1)).store(Vec(vraw))

            def st1(field, idx, vraw):
                primitive.ptr_store(vraw, fx.add_offset(field.ptr, fx.make_int_tuple(idx)))
            for kk in range_constexpr(N):
                for i in range_constexpr(NA):
                    st4(lds.A_lds, (i * N + kk) * 256 + lane * fx.Int32(4), gload4(av, i * N + kk))
                    st1(lds.SA_lds, (i * N + kk) * 64 + lane, gload1(sa, i * N + kk))
                for j in range_constexpr(NB):
                    st4(lds.B_lds, (j * N + kk) * 256 + lane * fx.Int32(4), gload4(bv, j * N + kk))
                    st1(lds.SB_lds, (j * N + kk) * 64 + lane, gload1(sb, j * N + kk))
            _llvm.inline_asm(ir.Type.parse("!llvm.void"), [], "s_waitcnt lgkmcnt(0)\ns_barrier", "", has_side_effects=True)

            def adr(field, eoff):
                return _raw(fx.ptrtoint(fx.add_offset(field.ptr, fx.make_int_tuple(eoff))))

            tyS = ir.Type.parse(ststrS)
            acc = [zero] * NACC
            for kk in range_constexpr(N):
                a_ad = [adr(lds.A_lds, (i * N + kk) * 256 + lane * fx.Int32(4)) for i in range_constexpr(NA)]
                b_ad = [adr(lds.B_lds, (j * N + kk) * 256 + lane * fx.Int32(4)) for j in range_constexpr(NB)]
                sa_ad = [adr(lds.SA_lds, (i * N + kk) * 64 + lane) for i in range_constexpr(NA)]
                sb_ad = [adr(lds.SB_lds, (j * N + kk) * 64 + lane) for j in range_constexpr(NB)]
                addrs = [_raw(x) for x in a_ad] + [_raw(x) for x in b_ad] + [_raw(x) for x in sa_ad] + [_raw(x) for x in sb_ad]
                ins = addrs + [_raw(acc[q]) for q in range_constexpr(NACC)]
                r = _llvm.inline_asm(tyS, ins, asmS, consS, has_side_effects=True)
                acc = [_llvm.extractvalue(res4, r, [q]) for q in range_constexpr(NACC)]
            outc = [Vec(acc[q]) for q in range_constexpr(NACC)]

        bo = buffer_ops.extract_base_index(out, address_space=1)
        base = _llvm.inttoptr(ir.Type.parse("!llvm.ptr<1>"), _raw(fx.Int64(bo)))
        for q in range_constexpr(NACC):
            for i in range_constexpr(4):
                off = ((fx.Int32(q) * fx.Int32(64) + lane) * fx.Int32(4) + fx.Int32(i)) * fx.Int32(4)
                _llvm.StoreOp(_raw(outc[q][i]), buffer_ops.get_element_ptr(base, byte_offset=_raw(off), elem_type=T.i8))

    @flyc.jit
    def launch(out: fx.Tensor, av: fx.Tensor, bv: fx.Tensor, sa: fx.Tensor, sb: fx.Tensor, stream: fx.Stream):
        k(out, av, bv, sa, sb).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)
    return launch


def main():
    d = "cuda"; torch.manual_seed(0)
    av = torch.randint(0, 2**31, (NA * N * 256,), dtype=torch.int32, device=d)
    bv = torch.randint(0, 2**31, (NB * N * 256,), dtype=torch.int32, device=d)
    eb_a = torch.randint(120, 135, (NA * N * 64,), dtype=torch.int32, device=d)
    eb_b = torch.randint(120, 135, (NB * N * 64,), dtype=torch.int32, device=d)
    sa = (eb_a * 0x01010101).to(torch.int32); sb = (eb_b * 0x01010101).to(torch.int32)
    stm = torch.cuda.current_stream(); res = {}
    for mode in ("ref", "asm"):
        out = torch.zeros(NACC * 64 * 4, dtype=torch.float32, device=d)
        cc = flyc.compile(build(mode), out, av, bv, sa, sb, stm)
        out.zero_(); cc(out, av, bv, sa, sb, stm); torch.cuda.synchronize(); res[mode] = out.clone()
    diff = (res["ref"] - res["asm"]).abs().max().item()
    rel = diff / (res["ref"].abs().max().item() + 1e-9)
    print(f"N={N} NACC={NACC} max|ref-asm|={diff:.3g} rel={rel:.3g} match={rel<1e-5} ref[:4]={res['ref'][:4].tolist()}")


main()
