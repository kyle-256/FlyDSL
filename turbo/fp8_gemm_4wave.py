# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""4-wave FP8 matmul with row-wise scaling for AMD CDNA4.

Algorithm derived from HipKittens FP8_4wave
(https://github.com/HazyResearch/HipKittens/blob/7782744ba1fd259a377a99e2ea8f71384cc80e55/kernels/gemm/fp8fp32/FP8_4wave/4_wave.cu#L1).

Global IO, scale loads, and bf16 stores go through the layout API
(``fx.rocdl.make_buffer_tensor`` + ``fx.copy`` with ``BufferCopyLDS128b``
/ ``BufferCopy{16,32,128}b``). MFMAs use ``fly.mma_atom_call_ssa`` so
the chained Vec(4, f32) accumulator stays on AGPR. The XOR swizzle and
the 8-buffer LDS pipeline ping-pong are kept as direct arithmetic to
preserve the original kernel's interleaved-cluster scheduling.

Optional B preshuffle uses the same on-disk layout as
``preshuffle_gemm_v2`` / ``shuffle_weight((16, 16))``.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import fly as fly_dialect
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import arith, buffer_ops, const_expr, range_constexpr
from flydsl.expr.arith import _to_raw as _raw
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import ArithValue
from kernels.fp8_gemm_utils import (
    mask_a_tail,
    G2SLoader,
    S2RLoader,
    StoreC,
    ceildiv,
    compute_global_swizzle,
    divmod,
    make_fp8_buffer_tensor,
    pack_i32x4_i32x8,
    swizzle_128,
    wait_barrier,
)


def asm_mma_do(a, b, c, mode="2", cbsz=0, blgp=0):
    """fp8 16x16x128 MFMA via inline asm, pinning the dst register class to
    avoid the v_accvgpr_read/write shuffle the mma_atom path emits (~2.3/MFMA).
    mode "2" (=a,v,v,0): accumulator AGPR in-place (D=C=$0); "3" (=v,v,v,0) VGPR in-place."""
    v4f32 = ir.VectorType.get([4], ir.F32Type.get())
    cons = {"2": "=a,v,v,0", "3": "=v,v,v,0"}.get(str(mode), "=&v,v,v,0")
    mods = f" cbsz:{cbsz} blgp:{blgp}" if (cbsz or blgp) else ""
    op = _llvm.InlineAsmOp(
        res=v4f32,
        operands_=[_raw(a), _raw(b), _raw(c)],
        asm_string=f"v_mfma_f32_16x16x128_f8f6f4 $0, $1, $2, $0{mods}",
        constraints=cons,
        has_side_effects=False,
    )
    return Vec(op.result)


class Mfma16x16x128:
    """16x16x128 fp8 MFMA. asm_mode 0 = mma_atom (default, emits accvgpr shuffle);
    2 = AGPR in-place inline-asm (no shuffle); 3 = VGPR in-place inline-asm."""

    def __init__(self, n_tiles_a, n_tiles_b, asm_mode=2):
        self.atom = fx.make_mma_atom(fx.rocdl.cdna4.MFMA_Scale(16, 16, 128, fx.Float8E4M3FN))
        self.accum_type = Vec.make_type(4, fx.Float32)
        self.zero_value = Vec.filled(4, 0.0, fx.Float32)
        self.n_tiles_a = n_tiles_a
        self.n_tiles_b = n_tiles_b
        self.asm_mode = asm_mode

    def idx(self, i, j):
        return i * self.n_tiles_b + j

    def _do_mma(self, a, b, c):
        if const_expr(self.asm_mode != 0):
            return asm_mma_do(a, b, c, mode=str(self.asm_mode))
        return fly_dialect.mma_atom_call_ssa([self.accum_type], self.atom, a, b, c)

    def call(self, a, b, c):
        assert len(a) == self.n_tiles_a
        assert len(b) == self.n_tiles_b
        assert len(c) == self.n_tiles_a * self.n_tiles_b
        for i in range_constexpr(self.n_tiles_a):
            for j in range_constexpr(self.n_tiles_b):
                c[self.idx(i, j)] = self._do_mma(a[i], b[j], c[self.idx(i, j)])
        return c

    def call_one(self, a, b, c, i, j):
        assert i < self.n_tiles_a and j < self.n_tiles_b
        return self._do_mma(a[i], b[j], c[self.idx(i, j)])


class StoreCScalar:
    """Tensorwise (scalar) output dequant store: out = (acc * scale).to(bf16),
    scale = a_scale * b_scale (both scalar)."""

    def __init__(self, scale, C, c_rows, c_cols, c_idx_fn, n_tiles_a, n_tiles_b):
        self.c_rows = c_rows
        self.c_cols = c_cols
        self.lane_id = fx.thread_idx.x % 64
        self.c_idx_fn = c_idx_fn
        self.n_tiles_a = n_tiles_a
        self.n_tiles_b = n_tiles_b
        self.scale = scale
        c_nbytes = c_rows * c_cols * 2
        gC = fx.rocdl.make_buffer_tensor(C, max_size=False, num_records_bytes=c_nbytes)
        self.c_div = fx.logical_divide(gC, fx.make_layout(1, 1))
        self.out_atom_1 = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), fx.BFloat16)
        self.reg_bf16_1 = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.BFloat16)

    def _store_bf16(self, value_bf16, c_index):
        fx.memref_store_vec(Vec.filled(1, value_bf16, fx.BFloat16), self.reg_bf16_1)
        fx.copy(self.out_atom_1, self.reg_bf16_1, fx.slice(self.c_div, (None, fx.Int32(c_index))))

    def store(self, c_frag, base_row, base_col):
        for ti in range_constexpr(self.n_tiles_a):
            row = base_row + ti * 16 + (self.lane_id // 16) * 4
            for tj in range_constexpr(self.n_tiles_b):
                col = base_col + tj * 16 + self.lane_id % 16
                col_valid = col < self.c_cols
                oob = fx.Int32(self.c_rows * self.c_cols)
                vec_f32 = Vec(c_frag[self.c_idx_fn(ti, tj)])
                for i in range_constexpr(4):
                    scaled = (vec_f32[i] * self.scale).to(fx.BFloat16)
                    c_index = (row + i) * self.c_cols + col
                    self._store_bf16(scaled, arith.select(col_valid, c_index, oob))


def _min(a, b):
    return arith.select(a < b, a, b)


def _xcd_swizzle(num_pid_m, num_pid_n):
    NUM_XCDS = 8
    WGM = 4
    NUM_CUS = 32 * NUM_XCDS
    SWIZZLE_THRESHOLD = 4 * NUM_CUS

    wgid = fx.block_idx.x

    num_wg = num_pid_m * num_pid_n

    # Simple path: no XCD remapping.
    simple_m, simple_n = divmod(wgid, num_pid_n)

    # XCD-remapped path.
    intra_xcd, xcd = divmod(wgid, NUM_XCDS)
    wgid_remap = xcd * (num_wg // NUM_XCDS) + intra_xcd
    num_wgid_in_group = WGM * num_pid_n
    group_id, intra_group = divmod(wgid_remap, num_wgid_in_group)
    first_pid_m = group_id * WGM
    group_size_m = _min(num_pid_m - first_pid_m, WGM)
    pid_n, intra_group_m = divmod(intra_group, group_size_m)
    pid_m = first_pid_m + intra_group_m

    use_simple = (num_wg <= SWIZZLE_THRESHOLD) | (num_wg % NUM_XCDS != 0)
    return (arith.select(use_simple, simple_m, pid_m), arith.select(use_simple, simple_n, pid_n))


def compile_fp8_gemm_4w(
    *,
    K: int,
    BLOCK_M: int = 256,
    BLOCK_N: int = 256,
    use_xcd_remap: bool = True,
    b_preshuffled: bool = False,
    waves_per_eu: int = 1,
    agpr_alloc: int = 0,  # 0=auto (asm modes force 16*N_ACCUMS AGPRs, required for correctness); N>0 force "N,N"; -N allow "0,N"
    asm_mma: int = 2,  # 2=AGPR in-place inline-asm MFMA (kills accvgpr shuffle, default); 0=atom; 3=VGPR in-place
    fuse_q: bool = False,  # True: A/B are BF16; tensorwise-quant cast (x*inv_scale->e4m3 via cvt_pk_fp8_f32) fused into G2S; A_scale/B_scale are scalar quant scales. CORRECT (SNR 55) but ~1.5-2x SLOWER than separate quant+gemm: inline cvt VALU doesn't overlap MFMA, so the dedicated peak-BW quant kernel wins. Kept as reference; fusion is net-negative here.
    tw_scalar: bool = False,  # True: fp8 inputs (NOT fused) + tensorwise scalar dequant store (A_scale/B_scale scalar). Production tensorwise dense/grouped path.
):
    # MFMA atom is 16x16x128; 4 waves in a 2x2 config require BLOCK >= 64.
    BLOCK_K = 128
    LDS_BLOCK_M = BLOCK_M // 2
    LDS_BLOCK_N = BLOCK_N // 2

    assert BLOCK_M >= 64 and BLOCK_M % 64 == 0 and BLOCK_N >= 64 and BLOCK_N % 64 == 0
    # Native K-tail: ceil(K/128) iters; final block's invalid K-cols (>=K_TAIL)
    # zeroed on A via mask_a_tail. One kernel handles any K (no sub-kernel dispatch).
    K_ITERS = (K + BLOCK_K - 1) // BLOCK_K
    K_TAIL = K % BLOCK_K
    assert K_ITERS >= 2, f"need K>=129 (ceil(K/128)>=2), got K_ITERS={K_ITERS}"
    # Number of 16-row 16x128 tiles per wave per A/B partition.
    N_TILES_A = BLOCK_M // 4 // 16
    N_TILES_B = BLOCK_N // 4 // 16
    N_ACCUMS = N_TILES_A * N_TILES_B
    assert N_ACCUMS > 0
    # asm in-place MFMA needs the AGPR file sized to the accumulators (4 quadrants x
    # N_ACCUMS vec4 x 4 AGPR); without the hint the allocator under-provisions AGPR and
    # the in-place accumulators alias -> wrong result. Auto unless caller overrides.
    _agpr_eff = agpr_alloc if agpr_alloc != 0 else (16 * N_ACCUMS if asm_mma != 0 else 0)

    N_LDS_ROUNDS = max(N_TILES_A, N_TILES_B)

    _use_interleaved_block = BLOCK_M == 256 and BLOCK_N == 256

    a_lds_size = LDS_BLOCK_M * BLOCK_K
    b_lds_size = LDS_BLOCK_N * BLOCK_K

    @fx.struct
    class SharedStorage:
        A_lds_cur_0: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        A_lds_cur_1: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        A_lds_next_0: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        A_lds_next_1: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        B_lds_cur_0: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
        B_lds_cur_1: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
        B_lds_next_0: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
        B_lds_next_1: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]

    @flyc.kernel
    def kernel_gemm(
        A: fx.Tensor, B_T: fx.Tensor, C: fx.Tensor, A_scale: fx.Tensor, B_scale: fx.Tensor, c_m: fx.Int32, c_n: fx.Int32
    ):
        F8_IR_t = fx.Float8E4M3FN.ir_type

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        a_cur0 = lds.A_lds_cur_0
        a_cur1 = lds.A_lds_cur_1
        a_next0 = lds.A_lds_next_0
        a_next1 = lds.A_lds_next_1
        b_cur0 = lds.B_lds_cur_0
        b_cur1 = lds.B_lds_cur_1
        b_next0 = lds.B_lds_next_0
        b_next1 = lds.B_lds_next_1

        lane_id = fx.thread_idx.x % 64
        wave_id = fx.thread_idx.x // 64

        n_blocks = ceildiv(c_n, BLOCK_N)
        if const_expr(use_xcd_remap):
            tile_i, tile_j = _xcd_swizzle(ceildiv(c_m, BLOCK_M), n_blocks)
        else:
            tile_i, tile_j = divmod(fx.block_idx.x, n_blocks)

        wave_i = wave_id // 2
        wave_j = wave_id % 2
        A0_gl_offset = (tile_i * BLOCK_M) * K
        A1_gl_offset = (tile_i * BLOCK_M + LDS_BLOCK_M) * K
        A_K_STEP = BLOCK_K
        B0_gl_offset = (tile_j * BLOCK_N) * K
        B1_gl_offset = (tile_j * BLOCK_N + LDS_BLOCK_N) * K
        B_K_STEP = (2 * 1024) if b_preshuffled else BLOCK_K

        if const_expr(fuse_q):
            # BF16 inputs; fused quant cast in G2S. Read scalar tensorwise scales.
            a_rsrc = buffer_ops.create_buffer_resource(A, max_size=False, num_records_bytes=c_m * K * 2)
            b_rsrc = buffer_ops.create_buffer_resource(B_T, max_size=False, num_records_bytes=c_n * K * 2)
            sa_rsrc = buffer_ops.create_buffer_resource(A_scale, max_size=False, num_records_bytes=4)
            sb_rsrc = buffer_ops.create_buffer_resource(B_scale, max_size=False, num_records_bytes=4)
            a_scale_v = ArithValue(buffer_ops.buffer_load(sa_rsrc, fx.Int32(0), vec_width=1, dtype=fx.Float32.ir_type))
            b_scale_v = ArithValue(buffer_ops.buffer_load(sb_rsrc, fx.Int32(0), vec_width=1, dtype=fx.Float32.ir_type))
            inv_a = 1.0 / a_scale_v
            inv_b = 1.0 / b_scale_v
            out_scale = a_scale_v * b_scale_v
        else:
            gA = make_fp8_buffer_tensor(A, F8_IR_t)
            gB = make_fp8_buffer_tensor(B_T, F8_IR_t)
            ga_div = fx.logical_divide(gA, fx.make_layout(1, 1))
            gb_div = fx.logical_divide(gB, fx.make_layout(1, 1))
            if const_expr(tw_scalar):
                # tensorwise scalar dequant: out_scale = a_scale * b_scale (both scalar)
                sa_rsrc = buffer_ops.create_buffer_resource(A_scale, max_size=False, num_records_bytes=4)
                sb_rsrc = buffer_ops.create_buffer_resource(B_scale, max_size=False, num_records_bytes=4)
                a_scale_v = ArithValue(buffer_ops.buffer_load(sa_rsrc, fx.Int32(0), vec_width=1, dtype=fx.Float32.ir_type))
                b_scale_v = ArithValue(buffer_ops.buffer_load(sb_rsrc, fx.Int32(0), vec_width=1, dtype=fx.Float32.ir_type))
                out_scale = a_scale_v * b_scale_v

        def _compute_lds_swizzle(s2r, preshuffled=False):
            lds_swz = []
            for row_offset in range_constexpr(s2r.n_tiles):
                row = s2r.wave_idx * (s2r.n_tiles * 16) + row_offset * 16 + lane_id % 16
                swz = []
                for i in range_constexpr(2):
                    col = (lane_id // 16) * 16 + i * 64
                    if const_expr(preshuffled):
                        swz.append((row // 8) * 1024 + (row % 8) * 16 + (col // 16) * 128)
                    else:
                        r, c = swizzle_128(row, col)
                        swz.append(r * BLOCK_K + c)
                lds_swz.append(swz)
            return lds_swz

        mfma = Mfma16x16x128(N_TILES_A, N_TILES_B, asm_mode=asm_mma)

        def _interleaved_cluster(
            lds_dst,
            g2s,
            k_offset,
            s2r,
            lds_src,
            a,
            b,
            c,
            lds_src_preshuffled=False,
        ):
            rt_dst = []

            c[mfma.idx(0, 0)] = mfma.call_one(a, b, c, 0, 0)
            c[mfma.idx(0, 1)] = mfma.call_one(a, b, c, 0, 1)

            lds_swz = _compute_lds_swizzle(s2r, preshuffled=lds_src_preshuffled)
            g2s.load_one(lds_dst, k_offset, 0)
            rt_dst_0 = s2r.load_one(lds_src, lds_swz[0][0])

            c[mfma.idx(0, 2)] = mfma.call_one(a, b, c, 0, 2)

            rt_dst_1 = s2r.load_one(lds_src, lds_swz[0][1])
            rt_dst.append(pack_i32x4_i32x8(rt_dst_0, rt_dst_1))

            c[mfma.idx(0, 3)] = mfma.call_one(a, b, c, 0, 3)

            g2s.load_one(lds_dst, k_offset, 1)
            rt_dst_0 = s2r.load_one(lds_src, lds_swz[1][0])

            c[mfma.idx(1, 0)] = mfma.call_one(a, b, c, 1, 0)
            c[mfma.idx(1, 1)] = mfma.call_one(a, b, c, 1, 1)

            rt_dst_1 = s2r.load_one(lds_src, lds_swz[1][1])
            rt_dst.append(pack_i32x4_i32x8(rt_dst_0, rt_dst_1))

            c[mfma.idx(1, 2)] = mfma.call_one(a, b, c, 1, 2)
            c[mfma.idx(1, 3)] = mfma.call_one(a, b, c, 1, 3)

            g2s.load_one(lds_dst, k_offset, 2)
            rt_dst_0 = s2r.load_one(lds_src, lds_swz[2][0])

            c[mfma.idx(2, 0)] = mfma.call_one(a, b, c, 2, 0)
            c[mfma.idx(2, 1)] = mfma.call_one(a, b, c, 2, 1)

            rt_dst_1 = s2r.load_one(lds_src, lds_swz[2][1])
            rt_dst.append(pack_i32x4_i32x8(rt_dst_0, rt_dst_1))

            c[mfma.idx(2, 2)] = mfma.call_one(a, b, c, 2, 2)
            c[mfma.idx(2, 3)] = mfma.call_one(a, b, c, 2, 3)

            g2s.load_one(lds_dst, k_offset, 3)
            rt_dst_0 = s2r.load_one(lds_src, lds_swz[3][0])

            c[mfma.idx(3, 0)] = mfma.call_one(a, b, c, 3, 0)
            c[mfma.idx(3, 1)] = mfma.call_one(a, b, c, 3, 1)

            rt_dst_1 = s2r.load_one(lds_src, lds_swz[3][1])
            rt_dst.append(pack_i32x4_i32x8(rt_dst_0, rt_dst_1))

            c[mfma.idx(3, 2)] = mfma.call_one(a, b, c, 3, 2)
            c[mfma.idx(3, 3)] = mfma.call_one(a, b, c, 3, 3)

            return c, rt_dst

        def _compute_cluster(
            lds_dst,
            g2s,
            k_offset,
            s2r,
            lds_src,
            a,
            b,
            c,
            lds_src_preshuffled=False,
        ):
            g2s.load(lds_dst, k_offset)
            rt_dst = s2r.load(lds_src, preshuffled=lds_src_preshuffled)
            c = mfma.call(a, b, c)
            return c, rt_dst

        def _compute_block(
            lds_dst,
            g2s,
            k_offset,
            s2r,
            lds_src,
            a,
            b,
            c,
            lds_src_preshuffled=False,
        ):
            if const_expr(_use_interleaved_block):
                return _interleaved_cluster(
                    lds_dst,
                    g2s,
                    k_offset,
                    s2r,
                    lds_src,
                    a,
                    b,
                    c,
                    lds_src_preshuffled=lds_src_preshuffled,
                )
            else:
                return _compute_cluster(
                    lds_dst,
                    g2s,
                    k_offset,
                    s2r,
                    lds_src,
                    a,
                    b,
                    c,
                    lds_src_preshuffled=lds_src_preshuffled,
                )

        # Each wave handles 2x2 64x64 sub-tiles of the output.
        c00_frag = [mfma.zero_value] * N_ACCUMS
        c01_frag = [mfma.zero_value] * N_ACCUMS
        c10_frag = [mfma.zero_value] * N_ACCUMS
        c11_frag = [mfma.zero_value] * N_ACCUMS

        gl_off_a = compute_global_swizzle(lane_id, wave_id, K, N_LDS_ROUNDS, preshuffled=False)
        gl_off_b = compute_global_swizzle(lane_id, wave_id, K, N_LDS_ROUNDS, preshuffled=b_preshuffled)

        class FusedQuantG2SLoader:
            """Like G2SLoader but reads a BF16 global tensor, casts to FP8 in-register
            (x * inv_scale -> e4m3), and writes FP8 to LDS. Eliminates the separate
            dynamic-quant cast pass; the cast VALU overlaps the MFMA chain. ``inv_scale``
            is a runtime scalar (1 / tensorwise-quant-scale). Reads use the SAME swizzled
            global offsets and LDS destinations as G2SLoader, so S2R is unchanged."""

            def __init__(self, gl_rsrc, gl_offsets, n_load_steps, lds_dtype, wave_id, inv_scale):
                self.gl_rsrc = gl_rsrc  # bf16 buffer resource
                self.gl_offsets = gl_offsets
                self.n_load_steps = n_load_steps
                self.wave_id = wave_id
                self.inv_scale = inv_scale
                self.n_waves = fx.block_dim.x // 64
                self.LdsPtr_t = fx.PointerType.get(lds_dtype, 2, 512)

            def _load_cast_store(self, lds_dst, k_offset, step):
                off = self.gl_offsets[step] + k_offset  # element offset (bf16 == fp8 count)
                parts = []
                for j in range_constexpr(4):
                    v = buffer_ops.buffer_load(
                        self.gl_rsrc, fx.Int32(off + j * 4), vec_width=4, dtype=fx.BFloat16.ir_type
                    )
                    parts.append(Vec(v))
                v01 = parts[0].shuffle(parts[1], list(range(8)))
                v23 = parts[2].shuffle(parts[3], list(range(8)))
                v16 = v01.shuffle(v23, list(range(16)))  # 16 bf16
                f = v16.to(fx.Float32) * self.inv_scale  # Vec 16 f32
                # f32 -> e4m3 via hardware cvt_pk_fp8_f32 (2 f32 -> 2 fp8 per call) -> 4 i32 words,
                # written to LDS at base + lane*16 (matches buffer_load_lds dwordx4 lane layout).
                i32t = fx.Int32.ir_type
                c0 = fx.Int32(0)
                lane_id = fx.thread_idx.x % 64
                step_off = self.wave_id * 1024 + step * (self.n_waves * 1024)
                base = fx.Int32(fx.ptrtoint(lds_dst.ptr)) + fx.Int32(step_off) + lane_id * fx.Int32(16)
                I32Ptr_t = fx.PointerType.get(fx.Int32.ir_type, 2, 512)
                ws = []
                for w in range_constexpr(4):
                    pw = fx.rocdl.cvt_pk_fp8_f32(i32t, f[4 * w + 0], f[4 * w + 1], c0, 0)
                    pw = fx.rocdl.cvt_pk_fp8_f32(i32t, f[4 * w + 2], f[4 * w + 3], pw, 1)
                    ws.append(Vec.filled(1, fx.Int32(pw), fx.Int32))
                v4 = ws[0].shuffle(ws[1], [0, 1]).shuffle(ws[2].shuffle(ws[3], [0, 1]), [0, 1, 2, 3])
                view = fx.make_view(fx.inttoptr(I32Ptr_t, base), fx.make_layout(4, 1))
                fx.memref_store_vec(v4, view)

            def load(self, lds_dst, k_offset):
                for step in range_constexpr(self.n_load_steps):
                    self._load_cast_store(lds_dst, k_offset, step)

            def load_one(self, lds_dst, k_offset, step):
                self._load_cast_store(lds_dst, k_offset, step)

        if const_expr(fuse_q):
            a_g2s = FusedQuantG2SLoader(a_rsrc, gl_off_a, N_TILES_A, F8_IR_t, wave_id, inv_a)
            b_g2s = FusedQuantG2SLoader(b_rsrc, gl_off_b, N_TILES_B, F8_IR_t, wave_id, inv_b)
        else:
            a_g2s = G2SLoader(ga_div, gl_off_a, N_TILES_A, F8_IR_t, wave_id)
            b_g2s = G2SLoader(gb_div, gl_off_b, N_TILES_B, F8_IR_t, wave_id)
        a_s2r = S2RLoader(wave_i, N_TILES_A)
        b_s2r = S2RLoader(wave_j, N_TILES_B)
        if const_expr(fuse_q or tw_scalar):
            store_c = StoreCScalar(out_scale, C, c_m, c_n, mfma.idx, N_TILES_A, N_TILES_B)
        else:
            store_c = StoreC(A_scale, B_scale, C, c_m, c_n, mfma.idx, N_TILES_A, N_TILES_B)

        # Prologue: 8-buffer LDS pipeline pre-fill.
        a_g2s.load(a_cur0, A0_gl_offset + 0 * A_K_STEP)
        b_g2s.load(b_cur0, B0_gl_offset + 0 * B_K_STEP)
        b_g2s.load(b_cur1, B1_gl_offset + 0 * B_K_STEP)
        a_g2s.load(a_cur1, A1_gl_offset + 0 * A_K_STEP)

        a_g2s.load(a_next0, A0_gl_offset + 1 * A_K_STEP)
        b_g2s.load(b_next0, B0_gl_offset + 1 * B_K_STEP)
        b_g2s.load(b_next1, B1_gl_offset + 1 * B_K_STEP)
        a_g2s.load(a_next1, A1_gl_offset + 1 * A_K_STEP)

        wait_barrier((3 * N_TILES_A) + (4 * N_TILES_B))

        a0_frag = a_s2r.load(a_cur0)

        wait_barrier((3 * N_TILES_A) + (3 * N_TILES_B))

        b0_frag = b_s2r.load(b_cur0, preshuffled=b_preshuffled)

        for k in range_constexpr(K_ITERS - 2):
            wait_barrier((2 * N_TILES_A) + (2 * N_TILES_B))

            c00_frag, b1_frag = _compute_block(
                a_cur0,
                a_g2s,
                A0_gl_offset + (k + 2) * A_K_STEP,
                b_s2r,
                b_cur1,
                a0_frag,
                b0_frag,
                c00_frag,
                lds_src_preshuffled=b_preshuffled,
            )

            c01_frag, a1_frag = _compute_block(
                b_cur0,
                b_g2s,
                B0_gl_offset + (k + 2) * B_K_STEP,
                a_s2r,
                a_cur1,
                a0_frag,
                b1_frag,
                c01_frag,
            )

            wait_barrier((2 * N_TILES_A) + (2 * N_TILES_B))

            c10_frag, a0_frag = _compute_block(
                b_cur1,
                b_g2s,
                B1_gl_offset + (k + 2) * B_K_STEP,
                a_s2r,
                a_next0,
                a1_frag,
                b0_frag,
                c10_frag,
            )

            c11_frag, b0_frag = _compute_block(
                a_cur1,
                a_g2s,
                A1_gl_offset + (k + 2) * A_K_STEP,
                b_s2r,
                b_next0,
                a1_frag,
                b1_frag,
                c11_frag,
                lds_src_preshuffled=b_preshuffled,
            )

            a_cur0, a_next0 = a_next0, a_cur0
            a_cur1, a_next1 = a_next1, a_cur1
            b_cur0, b_next0 = b_next0, b_cur0
            b_cur1, b_next1 = b_next1, b_cur1

        # Tail step k_iters - 2.
        wait_barrier((2 * N_TILES_A) + (2 * N_TILES_B))
        b1_frag = b_s2r.load(b_cur1, preshuffled=b_preshuffled)
        c00_frag = mfma.call(a0_frag, b0_frag, c00_frag)
        a1_frag = a_s2r.load(a_cur1)
        c01_frag = mfma.call(a0_frag, b1_frag, c01_frag)
        wait_barrier((1 * N_TILES_A) + (1 * N_TILES_B))
        a0_frag = a_s2r.load(a_next0)
        c10_frag = mfma.call(a1_frag, b0_frag, c10_frag)
        b0_frag = b_s2r.load(b_next0, preshuffled=b_preshuffled)
        c11_frag = mfma.call(a1_frag, b1_frag, c11_frag)

        a_cur0, a_next0 = a_next0, a_cur0
        a_cur1, a_next1 = a_next1, a_cur1
        b_cur0, b_next0 = b_next0, b_cur0
        b_cur1, b_next1 = b_next1, b_cur1

        # Tail step k_iters - 1.
        base_row = tile_i * BLOCK_M + wave_i * (N_TILES_A * 16)
        base_col = tile_j * BLOCK_N + wave_j * (N_TILES_B * 16)
        wait_barrier(0)
        b1_frag = b_s2r.load(b_cur1, preshuffled=b_preshuffled)
        a1_frag = a_s2r.load(a_cur1)
        # final K-block = the tail: zero A's invalid K-columns (>= K_TAIL) so they
        # contribute 0 to the mfma (no-op when K_TAIL==0).
        a0_frag = mask_a_tail(a0_frag, lane_id, K_TAIL)
        a1_frag = mask_a_tail(a1_frag, lane_id, K_TAIL)
        c00_frag = mfma.call(a0_frag, b0_frag, c00_frag)
        c01_frag = mfma.call(a0_frag, b1_frag, c01_frag)
        c10_frag = mfma.call(a1_frag, b0_frag, c10_frag)
        c11_frag = mfma.call(a1_frag, b1_frag, c11_frag)

        store_c.store(c00_frag, base_row + 0, base_col + 0)
        store_c.store(c01_frag, base_row + 0, base_col + LDS_BLOCK_N)
        store_c.store(c10_frag, base_row + LDS_BLOCK_M, base_col + 0)
        store_c.store(c11_frag, base_row + LDS_BLOCK_M, base_col + LDS_BLOCK_N)

    @flyc.jit
    def launch_gemm(
        A: fx.Tensor,
        B_T: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        c_m: fx.Int32,
        c_n: fx.Int32,
        stream: fx.Stream,
    ):
        grid_x = ceildiv(c_m, BLOCK_M) * ceildiv(c_n, BLOCK_N)
        _attrs = {"rocdl.waves_per_eu": waves_per_eu, "rocdl.flat_work_group_size": "256,256"}
        if _agpr_eff != 0:
            _alloc = f"0,{-_agpr_eff}" if _agpr_eff < 0 else f"{_agpr_eff},{_agpr_eff}"
            _attrs["passthrough"] = [["amdgpu-agpr-alloc", _alloc]]
        kernel_gemm(
            A,
            B_T,
            C,
            A_scale,
            B_scale,
            c_m,
            c_n,
            value_attrs=_attrs,
        ).launch(grid=(grid_x, 1, 1), block=(256, 1, 1), stream=stream)

    return launch_gemm
