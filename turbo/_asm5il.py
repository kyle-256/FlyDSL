"""Step 5b: TRUE software-pipeline (the >pipe lever). Double-buffered fragments:
preamble reads buffer0 (iter0); the unrolled-by-2 loop, each half, ISSUES the
next iter's NA+NB ds_reads into the OTHER buffer (they proceed async over LGKM)
then `s_waitcnt lgkmcnt(NA+NB)` (FIFO: leaving the just-issued batch in flight ==
the previous batch -- the CURRENT buffer -- has completed) then runs the current
buffer's MFMAs. So iter k+1's LDS reads overlap iter k's MFMAs == latency hidden,
unlike step-5's serial read->waitcnt(0)->mfma.

LDS gets +2 K-block slack: the last two prefetches read past the real K range into
uninitialized LDS (garbage, never fed to an MFMA), which is in-bounds so no fault.
Validated bit-exact vs an in-order intrinsic reference.
"""
import sys, os, torch
ACC_AGPR = os.environ.get("ACC_AGPR", "0") == "1"   # acc output to AGPR ("=a") vs VGPR ("=v")
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


N = 8          # K-blocks (must be even)
NA = 2
NB = 2
NACC = NA * NB
NSLACK = 2     # extra LDS blocks for trailing prefetch over-read
BLK = 1024
MFMA = "v_mfma_scale_f32_16x16x128_f8f6f4 ${d}, ${a}, ${b}, ${d}, ${sa}, ${sb} op_sel_hi:[0,0,0] cbsz:4 blgp:4"


def _layout():
    """Outputs: acc_q(NACC), fragA[2][NA]+fragB[2][NB] (2*(NA+NB) vec4i32),
    cnt, addr_a(NA)+addr_b(NB). Inputs: a_init(NA),b_init(NB),sa,sb,half,stride,
    zeros(NACC)."""
    nf = NA + NB
    o_acc = list(range(NACC))
    base = NACC
    o_fa = [[base + b * nf + i for i in range(NA)] for b in range(2)]
    o_fb = [[base + b * nf + NA + j for j in range(NB)] for b in range(2)]
    base += 2 * nf
    o_cnt = base; base += 1
    o_aa = [base + i for i in range(NA)]; base += NA
    o_ba = [base + j for j in range(NB)]; base += NB
    n_out = base
    i_sa = n_out + nf; i_sb = i_sa + 1; i_half = i_sa + 2; i_stride = i_sa + 3

    out_types = ["vector<4xf32>"] * NACC + ["vector<4xi32>"] * (2 * nf) + ["i32"] + ["i32"] * nf
    st_str = "!llvm.struct<(" + ", ".join(out_types) + ")>"
    acc_c = "=a" if ACC_AGPR else "=v"
    out_cons = [acc_c] * NACC + ["=&v"] * (2 * nf) + ["=&s"] + ["=v"] * nf
    in_cons = [str(x) for x in o_aa] + [str(x) for x in o_ba] + ["v", "v", "s", "s"] + [str(q) for q in o_acc]
    cons = ",".join(out_cons + in_cons)

    def rd(buf):  # issue NA+NB ds_reads into buffer `buf`, then advance addrs
        s = []
        for i in range(NA): s.append(f"ds_read_b128 ${o_fa[buf][i]}, ${o_aa[i]}")
        for j in range(NB): s.append(f"ds_read_b128 ${o_fb[buf][j]}, ${o_ba[j]}")
        for i in range(NA): s.append(f"v_add_u32 ${o_aa[i]}, ${o_aa[i]}, ${i_stride}")
        for j in range(NB): s.append(f"v_add_u32 ${o_ba[j]}, ${o_ba[j]}, ${i_stride}")
        return s

    def mm(buf):  # MFMAs consuming buffer `buf`
        s = []
        for i in range(NA):
            for j in range(NB):
                s.append(MFMA.format(d=o_acc[i * NB + j], a=o_fa[buf][i], b=o_fb[buf][j], sa=i_sa, sb=i_sb))
        return s

    L = []
    L += rd(0)                                  # preamble: iter0 -> buffer0
    L.append(f"s_mov_b32 ${o_cnt}, 0")
    L.append("1:")
    L += rd(1)                                   # prefetch -> buffer1
    L.append(f"s_waitcnt lgkmcnt({nf})")         # buffer0 complete
    L += mm(0)                                    # mfma buffer0
    L += rd(0)                                    # prefetch -> buffer0
    L.append(f"s_waitcnt lgkmcnt({nf})")         # buffer1 complete
    L += mm(1)                                    # mfma buffer1
    L.append(f"s_add_u32 ${o_cnt}, ${o_cnt}, 1")
    L.append(f"s_cmp_lt_u32 ${o_cnt}, ${i_half}")
    L.append("s_cbranch_scc1 1b")
    return "\n".join(L), cons, st_str


def build(mode):
    asm, cons, st_str = _layout()
    NBLK = N + NSLACK
    NTI = NA * NBLK * 256; NTB = NB * NBLK * 256

    @fx.struct
    class Storage:
        A_lds: fx.Array[fx.Int32, NTI, 16]
        B_lds: fx.Array[fx.Int32, NTB, 16]

    @flyc.kernel(name=f"asm5il_{mode}", known_block_size=[64, 1, 1])
    def k(out: fx.Tensor, av: fx.Tensor, bv: fx.Tensor, sav: fx.Int32, sbv: fx.Int32, half: fx.Int32, stride: fx.Int32):
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
            for kk in range_constexpr(N):  # fill only real N blocks (tile stride = NBLK)
                for i in range_constexpr(NA):
                    lds_store(lds.A_lds, (i * NBLK + kk) * 256 + lane * fx.Int32(4), gload(av, i * N + kk))
                for j in range_constexpr(NB):
                    lds_store(lds.B_lds, (j * NBLK + kk) * 256 + lane * fx.Int32(4), gload(bv, j * N + kk))
            _llvm.inline_asm(ir.Type.parse("!llvm.void"), [], "s_waitcnt lgkmcnt(0)\ns_barrier", "", has_side_effects=True)

            def addr(field, tile):
                pp = fx.add_offset(field.ptr, fx.make_int_tuple(tile * NBLK * 256 + lane * fx.Int32(4)))
                return _raw(fx.ptrtoint(pp))
            a_init = [addr(lds.A_lds, i) for i in range_constexpr(NA)]
            b_init = [addr(lds.B_lds, j) for j in range_constexpr(NB)]

            st = ir.Type.parse(st_str)
            ops = ([_raw(x) for x in a_init] + [_raw(x) for x in b_init]
                   + [_raw(sav), _raw(sbv), _raw(half), _raw(stride)] + [_raw(zero)] * NACC)
            r = _llvm.inline_asm(st, ops, asm, cons, has_side_effects=True)
            outc = [Vec(_llvm.extractvalue(res4, r, [q])) for q in range_constexpr(NACC)]

        bo = buffer_ops.extract_base_index(out, address_space=1)
        base = _llvm.inttoptr(ir.Type.parse("!llvm.ptr<1>"), _raw(fx.Int64(bo)))
        for q in range_constexpr(NACC):
            for i in range_constexpr(4):
                off = ((fx.Int32(q) * fx.Int32(64) + lane) * fx.Int32(4) + fx.Int32(i)) * fx.Int32(4)
                _llvm.StoreOp(_raw(outc[q][i]), buffer_ops.get_element_ptr(base, byte_offset=_raw(off), elem_type=T.i8))

    @flyc.jit
    def launch(out: fx.Tensor, av: fx.Tensor, bv: fx.Tensor, sav: fx.Int32, sbv: fx.Int32, half: fx.Int32, stride: fx.Int32, stream: fx.Stream):
        k(out, av, bv, sav, sbv, half, stride).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)
    return launch


def main():
    d = "cuda"; torch.manual_seed(0)
    av = torch.randint(0, 2**31, (NA * N * 256,), dtype=torch.int32, device=d)
    bv = torch.randint(0, 2**31, (NB * N * 256,), dtype=torch.int32, device=d)
    sa = 0x7f7f7f7f; stm = torch.cuda.current_stream(); res = {}
    for mode in ("ref", "il"):
        out = torch.zeros(NACC * 64 * 4, dtype=torch.float32, device=d)
        cc = flyc.compile(build(mode), out, av, bv, sa, sa, N // 2, BLK, stm)
        out.zero_(); cc(out, av, bv, sa, sa, N // 2, BLK, stm); torch.cuda.synchronize(); res[mode] = out.clone()
    diff = (res["ref"] - res["il"]).abs().max().item()
    print(f"N={N} NA={NA} NB={NB} NACC={NACC} max|ref-il|={diff:.3g} match={diff<1e-2} ref[:4]={res['ref'][:4].tolist()}")


main()
