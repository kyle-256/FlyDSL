# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""8-wave MXFP8 matmul (per-1x32 E8M0 block scaling) for AMD CDNA4 (gfx950).

Derived from ``kernels/fp8_gemm_8wave.py`` (tensorwise FP8). The structural
difference vs the tensorwise kernel:

  * tensorwise applies a single per-row (A) / per-col (B) FP32 scale in the
    epilogue, with the MFMA run un-scaled (identity scale operand).
  * mxfp8 carries a per-32-element-K-block E8M0 scale that MUST be fed to the
    ``v_mfma_scale_f32_16x16x128_f8f6f4`` instruction per K-iteration. The
    epilogue therefore becomes a plain FP32->BF16 store (all scaling already
    folded into the accumulator by the MMA).

Scale operand semantics (gfx950): the MMA takes one i32 scale per operand,
holding 4 packed E8M0 bytes -- one byte per 32-K block. A single
16x16x128 MFMA spans K=128 == 4 micro-blocks, so exactly one i32 scale per
(row/col tile, K-iteration).

Scale tensor layout expected by this kernel (passed pre-packed from host):
  A_scale: int32 [M, K // 128]   (each i32 == 4 consecutive E8M0 bytes of a row)
  B_scale: int32 [N, K // 128]
i.e. the raw uint8 E8M0 [DIM, K//32] viewed little-endian as int32.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import buffer_ops, const_expr, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from kernels.fp8_gemm_utils import (
    G2SLoader,
    S2RLoader,
    ceildiv,
    compute_global_swizzle,
    divmod,
    make_fp8_buffer_tensor,
    wait_barrier,
)


def preshuffle_scale(e8m0_u8, K, n_tiles):
    """Host-side E8M0 scale pre-shuffle for the mxfp8 8-wave kernel.

    ``n_tiles`` = the per-wave sub-tile fan-out the kernel loads together in one
    vectorized dword{n_tiles} (A: BLOCK_M//64, B: BLOCK_N//128).

    Input : uint8 [DIM, K//32] row-major E8M0 (DIM multiple of 16*n_tiles).
    Output: int32 [DIM//(16*n_tiles), K//128, 64, n_tiles] where
        SP[grp, k, lane, s] = broadcast( scale[grp*16*n_tiles + s*16 + lane%16,
                                               4k + lane//16] )
    so a wave reads its ``n_tiles`` sub-tile scales for (grp, k) as one coalesced
    vector load of ``n_tiles`` contiguous dwords per lane, directly usable as the
    MFMA scale operand (opsel==0 reads byte 0; broadcast => byte-position safe).
    Byte-packing the n_tiles into one dword regressed (extract ALU on the
    load->MFMA dep chain); the bottleneck is VMEM-unit occupancy, not traffic.
    """
    import torch

    DIM, Kb = e8m0_u8.shape
    assert Kb == K // 32 and K % 128 == 0
    assert DIM % (16 * n_tiles) == 0, f"DIM={DIM} must be multiple of {16 * n_tiles}"
    K128 = K // 128
    G = DIM // (16 * n_tiles)
    s = e8m0_u8.reshape(DIM, K128, 4)                       # [DIM, k, g]
    s = s.reshape(G, n_tiles, 16, K128, 4)                  # [grp, s, r, k, g]
    s = s.permute(0, 3, 4, 2, 1).contiguous()              # [grp, k, g, r, s]
    s = s.reshape(G, K128, 64, n_tiles).to(torch.int32)    # lane == g*16 + r
    sp = s | (s << 8) | (s << 16) | (s << 24)
    return sp.contiguous()


def preshuffle_scale_b_comb(e8m0_u8, K):
    """Combined-B E8M0 pre-shuffle: pack BOTH N-regions' (b0,b1) 4 sub-tiles for a
    wave into one dword{4}, so the kernel issues a single dwordx4 for all B scales
    per K-iter (vs two loads). Requires N % 256 == 0.

    A wave's 4 B sub-tiles sit at cols c+{0,16,128,144} (b0: 0,16; b1: 128,144),
    c = block_n*256 + wave_n*32. Output int32 [N//64, K//128, 64, 4]:
        SP[grp, k, lane, s] = broadcast( scale[c + OFF[s] + lane%16, 4k + lane//16] )
    grp = block_n*4 + wave_n;  OFF = [0,16,128,144].
    """
    import torch

    N, Kb = e8m0_u8.shape
    assert Kb == K // 32 and K % 128 == 0 and N % 256 == 0
    K128 = K // 128
    OFF = [0, 16, 128, 144]
    s = e8m0_u8.reshape(N // 256, 256, K128, 4)  # [nblk, col256, k, g]
    # col256 = wn*32 + OFF[si] + r  (bijection of 0..255)
    wn = torch.arange(4).view(4, 1, 1)
    si = torch.arange(4).view(1, 4, 1)
    r = torch.arange(16).view(1, 1, 16)
    off = torch.tensor(OFF).view(1, 4, 1)
    colidx = (wn * 32 + off + r).reshape(-1)  # [4*4*16] = [wn,si,r] flattened
    g = s[:, colidx, :, :].reshape(N // 256, 4, 4, 16, K128, 4)  # [nblk, wn, si, r, k, g]
    g = g.permute(0, 1, 4, 5, 3, 2).contiguous()  # [nblk, wn, k, g, r, si]
    g = g.reshape(N // 64, K128, 64, 4).to(torch.int32)  # grp=nblk*4+wn, lane=g*16+r
    sp = g | (g << 8) | (g << 16) | (g << 24)
    return sp.contiguous()


class ScaleBComb:
    """Combined B scale loader (pairs with ``preshuffle_scale_b_comb``).

    One dwordx4 per lane returns [s0,s1,s2,s3]; (s0,s1)=b0 sub-tiles, (s2,s3)=b1.
    """

    def __init__(self, sp_tensor, dim, K, const=False):
        self.K128 = K // 128
        self.const = const
        self.lane = fx.thread_idx.x % 64
        if not const:
            nbytes = (dim // 64) * self.K128 * 64 * 4 * 4  # int32 records
            self.rsrc = buffer_ops.create_buffer_resource(sp_tensor, max_size=False, num_records_bytes=nbytes)

    def load(self, base, k):
        """base: sb_base0 (b0 region col base). Returns 4 i32 (b0:0,1  b1:2,3)."""
        if self.const:
            return [None, None, None, None]
        grp = (base // 256) * 4 + (base % 256) // 32
        idx = ((grp * self.K128 + k) * 64 + self.lane) * 4
        v = Vec(buffer_ops.buffer_load(self.rsrc, idx, vec_width=4, dtype=T.i32))
        return [v[i].ir_value() for i in range_constexpr(4)]

    def load_halves(self, base, lds_block_n, k):
        """Uniform N-half interface (mirrors mxfp4 ScaleBRegion.load_halves):
        return (b0_scales, b1_scales) for K128 index k. Combined path packs both
        halves in one dwordx4 (b0=0,1 b1=2,3); lds_block_n unused here."""
        v = self.load(base, k)
        return v[0:2], v[2:4]


class MfmaScale16x16x128:
    """16x16x128 f8f6f4 MFMA with per-block E8M0 scale operands.

    Mirrors ``Mfma16x16x128`` but routes through the raw rocdl intrinsic so
    the (scale_a, scale_b) i32 operands can be supplied per call.
    """

    def __init__(self, n_tiles_a, n_tiles_b, const_scale=False):
        self.res_ty = Vec.make_type(4, fx.Float32)
        self.zero_value = Vec.filled(4, 0.0, fx.Float32)
        self.n_tiles_a = n_tiles_a
        self.n_tiles_b = n_tiles_b
        self.const_scale = const_scale  # diagnostic: feed 0x7F7F7F7F immediate, no loads

    def idx(self, i, j):
        return i * self.n_tiles_b + j

    def _do_mma(self, a, b, c, sa, sb):
        # operand order: a, b, c, cbsz, blgp, opsel_a, scale_a, opsel_b, scale_b
        # fp8 path: cbsz=blgp=0, opsel=0 (one i32 == 4 E8M0 covering K=128).
        if self.const_scale:
            sa = sb = 0x7F7F7F7F
        return rocdl.mfma_scale_f32_16x16x128_f8f6f4(
            self.res_ty,
            [a, b, c, 0, 0, 0, sa, 0, sb],
        )

    def call(self, a, b, c, sa, sb):
        assert len(a) == self.n_tiles_a
        assert len(b) == self.n_tiles_b
        assert len(c) == self.n_tiles_a * self.n_tiles_b
        assert len(sa) == self.n_tiles_a
        assert len(sb) == self.n_tiles_b

        for i in range_constexpr(self.n_tiles_a):
            for j in range_constexpr(self.n_tiles_b):
                c[self.idx(i, j)] = self._do_mma(a[i], b[j], c[self.idx(i, j)], sa[i], sb[j])
        return c

    def call_one(self, a, b, c, i, j, sa, sb):
        return self._do_mma(a[i], b[j], c[self.idx(i, j)], sa[i], sb[j])


class ScaleS2R:
    """Per-lane E8M0 scale loader for v_mfma_scale_f32_16x16x128 (preshuffled).

    The 16x16x128 MFMA distributes K=128 so lane ``(g, r)`` with
    ``g = lane//16`` (0..3) and ``r = lane%16`` holds the A/B data for matrix
    row/col ``r`` and the 32-K micro-block ``g``. With opsel==0 the hardware
    samples byte 0 of each lane's scale operand, so lane ``(g, r)`` just needs
    ``scale[r, 4k+g]`` in a register.

    To make that a single fully-coalesced dword load with no per-lane ALU, the
    host pre-shuffles the raw E8M0 [DIM, K//32] into

        SP[rt, k, lane] = broadcast_u8_to_u32( scale[rt*16 + lane%16, 4k + lane//16] )

    laid out int32 [DIM//16, K//128, 64]. For row-tile ``rt`` and K-iter ``k``
    the 64 lanes of a wave read 64 contiguous dwords. See ``preshuffle_scale``.
    """

    def __init__(self, sp_tensor, dim, K, n_tiles, const=False):
        self.K128 = K // 128
        self.n_tiles = n_tiles
        self.const = const  # diagnostic: skip loads, return placeholders
        self.group_span = 16 * n_tiles
        self.lane = fx.thread_idx.x % 64  # == (lane//16)*16 + lane%16
        if not const:
            nbytes = (dim // self.group_span) * self.K128 * 64 * n_tiles * 4  # int32 records
            self.rsrc = buffer_ops.create_buffer_resource(sp_tensor, max_size=False, num_records_bytes=nbytes)

    def load(self, base, k):
        """base: runtime global row/col base for this (region, wave). Returns n_tiles i32.

        One vectorized dword{n_tiles} load: the n_tiles sub-tile scales for this
        wave at (group, k) are contiguous per lane (see ``preshuffle_scale``).
        """
        if self.const:
            return [None] * self.n_tiles
        grp = base // self.group_span
        idx = ((grp * self.K128 + k) * 64 + self.lane) * self.n_tiles
        v = Vec(buffer_ops.buffer_load(self.rsrc, idx, vec_width=self.n_tiles, dtype=T.i32))
        return [v[i].ir_value() for i in range_constexpr(self.n_tiles)]


class StoreCPlain:
    """Plain FP32 accumulator -> BF16 store (no scaling; scales folded in MMA)."""

    def __init__(self, C, c_rows, c_cols, c_idx_fn, n_tiles_a, n_tiles_b):
        self.c_rows = c_rows
        self.c_cols = c_cols
        self.lane_id = fx.thread_idx.x % 64
        self.c_idx_fn = c_idx_fn
        self.n_tiles_a = n_tiles_a
        self.n_tiles_b = n_tiles_b
        c_nbytes = c_rows * c_cols * 2  # BFloat16
        gC = fx.rocdl.make_buffer_tensor(C, max_size=False, num_records_bytes=c_nbytes)
        self.c_div = fx.logical_divide(gC, fx.make_layout(1, 1))
        self.out_atom_1 = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), fx.BFloat16)
        self.reg_bf16_1 = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.BFloat16)

    def _store_bf16(self, value_bf16, c_index):
        fx.memref_store_vec(Vec.filled(1, value_bf16, fx.BFloat16), self.reg_bf16_1)
        fx.copy(self.out_atom_1, self.reg_bf16_1, fx.slice(self.c_div, (None, fx.Int32(c_index))))

    def store(self, c_frag, base_row, base_col):
        from flydsl.expr import arith

        for ti in range_constexpr(self.n_tiles_a):
            row = base_row + ti * 16 + (self.lane_id // 16) * 4
            for tj in range_constexpr(self.n_tiles_b):
                col = base_col + tj * 16 + self.lane_id % 16
                col_valid = col < self.c_cols
                oob = fx.Int32(self.c_rows * self.c_cols)
                vec_f32 = Vec(c_frag[self.c_idx_fn(ti, tj)])
                for i in range_constexpr(4):
                    val = vec_f32[i].to(fx.BFloat16)
                    c_index = (row + i) * self.c_cols + col
                    self._store_bf16(val, arith.select(col_valid, c_index, oob))


def compile_mxfp8_gemm_8w(
    *, K: int, BLOCK_M: int = 256, BLOCK_N: int = 256, b_preshuffled: bool = False, const_scale: bool = False, scale_once: bool = False
):
    BLOCK_K = 128

    assert BLOCK_M >= 128 and BLOCK_N >= 256 and BLOCK_M % 128 == 0 and BLOCK_N % 256 == 0
    assert K % BLOCK_K == 0

    K_ITERS = K // BLOCK_K

    N_TILES_A = BLOCK_M // 64
    N_TILES_B = BLOCK_N // 128
    N_ACCUMS = N_TILES_A * N_TILES_B
    assert N_ACCUMS > 0

    LDS_BLOCK_M = BLOCK_M // 2
    LDS_BLOCK_N = BLOCK_N // 2

    N_LDS_STEPS_A = LDS_BLOCK_M // 64
    N_LDS_STEPS_B = LDS_BLOCK_N // 64
    N_LDS_ROUNDS = max(N_LDS_STEPS_A, N_LDS_STEPS_B)

    a_lds_size = LDS_BLOCK_M * BLOCK_K
    b_lds_size = LDS_BLOCK_N * BLOCK_K

    # scale-tile fanout per MFMA wrapper call (A sub-tiles / B sub-tiles per wave).
    SA_TILES = N_TILES_A
    SB_TILES = N_TILES_B

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

    @flyc.kernel(known_block_size=[512, 1, 1])
    def kernel_gemm(
        A: fx.Tensor,
        B_T: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        c_m: fx.Int32,
        c_n: fx.Int32,
    ):
        F8_IR_t = fx.Float8E4M3FN.ir_type

        n_blocks = ceildiv(c_n, BLOCK_N)

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
        wave_m = wave_id // 4
        wave_n = wave_id % 4
        block_m, block_n = divmod(fx.block_idx.x, n_blocks)

        A0_gl_offset = (block_m * BLOCK_M) * K
        A1_gl_offset = (block_m * BLOCK_M + LDS_BLOCK_M) * K
        B_K_STEP = (2 * 1024) if b_preshuffled else BLOCK_K
        B0_gl_offset = (block_n * BLOCK_N) * K
        B1_gl_offset = (block_n * BLOCK_N + LDS_BLOCK_N) * K

        gA = make_fp8_buffer_tensor(A, F8_IR_t)
        gB = make_fp8_buffer_tensor(B_T, F8_IR_t)
        a_div = fx.logical_divide(gA, fx.make_layout(1, 1))
        b_div = fx.logical_divide(gB, fx.make_layout(1, 1))

        gl_off_a = compute_global_swizzle(lane_id, wave_id, K, N_LDS_ROUNDS, preshuffled=False)
        gl_off_b = compute_global_swizzle(lane_id, wave_id, K, N_LDS_ROUNDS, preshuffled=b_preshuffled)

        mfma = MfmaScale16x16x128(N_TILES_A, N_TILES_B, const_scale=const_scale)

        a_g2s = G2SLoader(a_div, gl_off_a, N_LDS_STEPS_A, F8_IR_t, wave_id)
        b_g2s = G2SLoader(b_div, gl_off_b, N_LDS_STEPS_B, F8_IR_t, wave_id)
        a_s2r = S2RLoader(wave_m, N_TILES_A)
        b_s2r = S2RLoader(wave_n, N_TILES_B)

        sa_s2r = ScaleS2R(A_scale, c_m, K, SA_TILES, const=const_scale)
        sb_s2r = ScaleBComb(B_scale, c_n, K, const=const_scale)  # one dwordx4 = b0+b1 scales
        store_c = StoreCPlain(C, c_m, c_n, mfma.idx, N_TILES_A, N_TILES_B)

        # Global row/col bases for the two M / N regions (region1 = +LDS half).
        wave_m_offset = wave_m * (N_TILES_A * 16)
        wave_n_offset = wave_n * (N_TILES_B * 16)
        sa_base0 = fx.Int32(block_m * BLOCK_M + wave_m_offset)
        sa_base1 = sa_base0 + fx.Int32(LDS_BLOCK_M)
        sb_base0 = fx.Int32(block_n * BLOCK_N + wave_n_offset)
        sb_base1 = sb_base0 + fx.Int32(LDS_BLOCK_N)

        # 2x2 config of accumulators
        c00_frag = [mfma.zero_value] * N_ACCUMS
        c01_frag = [mfma.zero_value] * N_ACCUMS
        c10_frag = [mfma.zero_value] * N_ACCUMS
        c11_frag = [mfma.zero_value] * N_ACCUMS

        b_g2s.load(b_cur0, B0_gl_offset + 0 * B_K_STEP)
        a_g2s.load(a_cur0, A0_gl_offset + 0 * BLOCK_K)
        b_g2s.load(b_cur1, B1_gl_offset + 0 * B_K_STEP)
        a_g2s.load(a_cur1, A1_gl_offset + 0 * BLOCK_K)

        if wave_m == 1:
            rocdl.s_barrier()

        wait_barrier(N_LDS_STEPS_A + N_LDS_STEPS_B)

        b_g2s.load(b_next0, B0_gl_offset + 1 * B_K_STEP)
        a_g2s.load(a_next0, A0_gl_offset + 1 * BLOCK_K)
        b_g2s.load(b_next1, B1_gl_offset + 1 * B_K_STEP)

        wait_barrier(N_LDS_STEPS_A + 2 * N_LDS_STEPS_B)

        # 1-deep scale prefetch (2-deep spills: V=256 maxed, register pressure
        # dominated the latency-hiding benefit). Pre-load k=0, prefetch k+1, scale
        # loads distributed across barrier sections.
        sa0 = sa_s2r.load(sa_base0, 0)
        sa1 = sa_s2r.load(sa_base1, 0)
        sb_all = sb_s2r.load(sb_base0, 0)
        sb0, sb1 = sb_all[0:2], sb_all[2:4]

        for k in range_constexpr(K_ITERS - 2):
            if const_expr(not scale_once):
                sa0n = sa_s2r.load(sa_base0, k + 1)

            b0_frag = b_s2r.load(b_cur0, preshuffled=b_preshuffled)
            a0_frag = a_s2r.load(a_cur0)
            a_g2s.load(a_next1, A1_gl_offset + (k + 1) * BLOCK_K)
            rocdl.s_barrier()

            rocdl.s_setprio(1)
            c00_frag = mfma.call(a0_frag, b0_frag, c00_frag, sa0, sb0)
            rocdl.s_setprio(0)
            rocdl.s_barrier()

            b1_frag = b_s2r.load(b_cur1, preshuffled=b_preshuffled)
            b_g2s.load(b_cur0, B0_gl_offset + (k + 2) * B_K_STEP)
            if const_expr(not scale_once):
                sb_alln = sb_s2r.load(sb_base0, k + 1)  # one dwordx4 = both B regions
            rocdl.s_barrier()

            rocdl.s_setprio(1)
            c01_frag = mfma.call(a0_frag, b1_frag, c01_frag, sa0, sb1)
            rocdl.s_setprio(0)
            rocdl.s_barrier()

            a1_frag = a_s2r.load(a_cur1)
            a_g2s.load(a_cur0, A0_gl_offset + (k + 2) * BLOCK_K)
            if const_expr(not scale_once):
                sa1n = sa_s2r.load(sa_base1, k + 1)
            rocdl.s_barrier()

            rocdl.s_setprio(1)
            c10_frag = mfma.call(a1_frag, b0_frag, c10_frag, sa1, sb0)
            rocdl.s_setprio(0)
            rocdl.s_barrier()

            b_g2s.load(b_cur1, B1_gl_offset + (k + 2) * B_K_STEP)
            wait_barrier(2 * N_LDS_STEPS_A + N_LDS_STEPS_B)

            rocdl.s_setprio(1)
            c11_frag = mfma.call(a1_frag, b1_frag, c11_frag, sa1, sb1)
            rocdl.s_setprio(0)
            rocdl.s_barrier()

            a_cur0, a_next0 = a_next0, a_cur0
            a_cur1, a_next1 = a_next1, a_cur1
            b_cur0, b_next0 = b_next0, b_cur0
            b_cur1, b_next1 = b_next1, b_cur1
            if const_expr(not scale_once):
                sa0, sa1 = sa0n, sa1n
                sb_all = sb_alln
                sb0, sb1 = sb_all[0:2], sb_all[2:4]

        # Step k = K_ITERS - 2 (sa*/sb* hold scales[K_ITERS-2]; prefetch last iter)
        k = K_ITERS - 2
        if const_expr(not scale_once):
            sa0n = sa_s2r.load(sa_base0, K_ITERS - 1)
            sa1n = sa_s2r.load(sa_base1, K_ITERS - 1)
            sb_alln = sb_s2r.load(sb_base0, K_ITERS - 1)

        b0_frag = b_s2r.load(b_cur0, preshuffled=b_preshuffled)
        a0_frag = a_s2r.load(a_cur0)
        rocdl.s_barrier()

        rocdl.s_setprio(1)
        c00_frag = mfma.call(a0_frag, b0_frag, c00_frag, sa0, sb0)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        b1_frag = b_s2r.load(b_cur1, preshuffled=b_preshuffled)
        rocdl.s_barrier()

        rocdl.s_setprio(1)
        c01_frag = mfma.call(a0_frag, b1_frag, c01_frag, sa0, sb1)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        a1_frag = a_s2r.load(a_cur1)
        a_g2s.load(a_next1, A1_gl_offset + (K_ITERS - 1) * BLOCK_K)
        rocdl.s_barrier()

        rocdl.s_setprio(1)
        c10_frag = mfma.call(a1_frag, b0_frag, c10_frag, sa1, sb0)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        b0_frag = b_s2r.load(b_next0, preshuffled=b_preshuffled)
        rocdl.s_barrier()

        rocdl.s_setprio(1)
        c11_frag = mfma.call(a1_frag, b1_frag, c11_frag, sa1, sb1)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        a_cur0, a_next0 = a_next0, a_cur0
        a_cur1, a_next1 = a_next1, a_cur1
        b_cur0, b_next0 = b_next0, b_cur0
        b_cur1, b_next1 = b_next1, b_cur1
        if const_expr(not scale_once):
            sa0, sa1 = sa0n, sa1n
            sb_all = sb_alln
            sb0, sb1 = sb_all[0:2], sb_all[2:4]

        # Step k = K_ITERS - 1 (sa*/sb* already hold scales[K_ITERS-1])
        k = K_ITERS - 1
        a0_frag = a_s2r.load(a_cur0)
        wait_barrier(0)

        rocdl.s_setprio(1)
        c00_frag = mfma.call(a0_frag, b0_frag, c00_frag, sa0, sb0)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        b1_frag = b_s2r.load(b_cur1, preshuffled=b_preshuffled)
        rocdl.s_barrier()

        rocdl.s_setprio(1)
        c01_frag = mfma.call(a0_frag, b1_frag, c01_frag, sa0, sb1)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        a1_frag = a_s2r.load(a_cur1)
        rocdl.s_barrier()

        rocdl.s_setprio(1)
        c10_frag = mfma.call(a1_frag, b0_frag, c10_frag, sa1, sb0)
        c11_frag = mfma.call(a1_frag, b1_frag, c11_frag, sa1, sb1)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        # Store back to gmem (no scaling)
        base_row = block_m * BLOCK_M + wave_m_offset
        base_col = block_n * BLOCK_N + wave_n_offset

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
        kernel_gemm(
            A,
            B_T,
            C,
            A_scale,
            B_scale,
            c_m,
            c_n,
            value_attrs={"rocdl.waves_per_eu": 2, "rocdl.flat_work_group_size": "512,512"},
        ).launch(grid=(grid_x, 1, 1), block=(512, 1, 1), stream=stream)

    return launch_gemm
