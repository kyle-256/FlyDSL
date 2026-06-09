# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""MXFP4 dense GEMM, 4-wave 2x2 layout (gfx950), derived from
``kernels/fp8_gemm_4wave.py`` (which keeps the 64-accumulator-AGPR + interleaved
cluster structure that the competitor's a4w4 Gluon kernel also uses).

Adapts the fp8 4-wave skeleton to MXFP4: per-1x32 E8M0 block scales fed to
``mfma_scale_f32_16x16x128_f8f6f4`` (cbsz=4/blgp=4 fp4 mode), fp4 packed 2/byte
data path (16B per lane per 128-K MFMA, i32x4 padded to i32x8), and a plain
FP32->BF16 epilogue (scales folded into the accumulator by the MMA).

M1 = simple staged (single-buffer, blunt barriers) to validate the 4-wave tiling
+ scale mapping. M2 will add the fp8_gemm_4wave software pipeline + interleaved
cluster; M3 the padded / transposed-operand LDS for bank-conflict removal.

Scales reuse mxfp8 ``ScaleS2R`` for BOTH A and B (row/col symmetric); host packs
with ``preshuffle_scale(e8m0, K, n_tiles)`` where n_tiles = BLOCK_{M,N}//4//16.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr, rocdl

from kernels.fp8_gemm_utils import (
    G2SLoader,
    ceildiv,
    divmod,
    make_fp8_buffer_tensor,
    wait_barrier,
)
from turbo.mxfp4_gemm_8wave import (
    MfmaScaleFp4,
    PaddedG2SLoader,
    S2RLoaderFp4,
    fp4_g2s_offsets,
    wait_barrier_lgkm,
)
from turbo.mxfp8_gemm_8wave import ScaleS2R, StoreCPlain


def _il_mma(mfma, quads, prefetch, ua, n_ta, n_tb):
    """One K-iter's 64 inline-asm MFMAs (4 quadrants x n_ta*n_tb) with the
    prefetch G2S load_one's spread 1-per-8-MFMA. Module-level (no closures) so
    FlyDSL's kernel AST transform doesn't choke. Mutates each quad's c list."""
    li = 0
    cnt = 0
    n_pf = len(prefetch)
    for cc, aa, bb, sca, scb in quads:
        for i in range_constexpr(n_ta):
            for j in range_constexpr(n_tb):
                cc[mfma.idx(i, j)] = mfma._do(aa[i][0], bb[j][0], cc[mfma.idx(i, j)], sca[i], scb[j], ua)
                cnt += 1
                if n_pf and cnt % 8 == 0 and li < n_pf:
                    g, ld, ko, st = prefetch[li]
                    g.load_one(ld, ko, st)
                    li += 1
    while li < n_pf:
        g, ld, ko, st = prefetch[li]
        g.load_one(ld, ko, st)
        li += 1


def compile_mxfp4_gemm_4w(
    *,
    K: int,
    BLOCK_M: int = 256,
    BLOCK_N: int = 256,
    padded: bool = False,
    pad_bytes: int = 16,
    asm_mfma: bool = False,
    interleave: bool = False,
    asm_se: bool = False,
    mode: str = "",
    block_k: int = 128,
):
    # block_k: logical fp4 contracted per K-iter (16x16x128 MFMA does 128 K, so a
    # K-iter spans N_SUB == block_k/128 sub-block MFMAs per accumulator). Only the
    # `pipe` mode honors block_k; il/default hardwire 128.
    BLOCK_K = block_k if mode == "pipe" else 128
    assert BLOCK_M >= 64 and BLOCK_M % 64 == 0 and BLOCK_N >= 64 and BLOCK_N % 64 == 0
    assert K % BLOCK_K == 0 and BLOCK_K % 128 == 0

    K_ITERS = K // BLOCK_K
    N_SUB = BLOCK_K // 128
    BPR = BLOCK_K // 2  # 64 packed-fp4 bytes per K-iter row
    KSTEP = BPR
    K2 = K // 2
    # 4 waves in a 2x2 grid; each wave owns N_TILES_A x N_TILES_B 16x16 tiles per quadrant.
    N_TILES_A = BLOCK_M // 4 // 16
    N_TILES_B = BLOCK_N // 4 // 16
    N_ACCUMS = N_TILES_A * N_TILES_B
    assert N_ACCUMS > 0
    LDS_BLOCK_M = BLOCK_M // 2
    LDS_BLOCK_N = BLOCK_N // 2

    # G2S row coverage per step = n_waves(=4) * (64 / (BPR/16)) rows.
    _ROWS_PER_STEP = 64 // (BPR // 16) * (256 // 64)
    N_LDS_STEPS_A = LDS_BLOCK_M // _ROWS_PER_STEP
    N_LDS_STEPS_B = LDS_BLOCK_N // _ROWS_PER_STEP
    LDS_ROW_STRIDE = BPR + (pad_bytes if padded else 0)
    a_lds_size = LDS_BLOCK_M * LDS_ROW_STRIDE
    b_lds_size = LDS_BLOCK_N * LDS_ROW_STRIDE

    @fx.struct
    class SharedStorage:
        A_lds0: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        A_lds1: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        B_lds0: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
        B_lds1: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]

    @fx.struct
    class SharedStoragePipe:
        A_lds_cur_0: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        A_lds_cur_1: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        A_lds_next_0: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        A_lds_next_1: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        B_lds_cur_0: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
        B_lds_cur_1: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
        B_lds_next_0: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
        B_lds_next_1: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]

    @flyc.kernel(known_block_size=[256, 1, 1])
    def kernel_gemm_il(
        A: fx.Tensor,
        B_T: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        c_m: fx.Int32,
        c_n: fx.Int32,
    ):
        # Interleaved: double-buffered; prefetch G2S(k+1) into next buffers spread
        # among iter-k's 64 inline-asm MFMAs (1 buffer_load_lds per 8 MFMAs), so
        # async gmem->LDS overlaps MFMA compute. asm MFMA (opaque, asm_se) blocks
        # the LLVM scheduler from re-clustering the MFMAs back together -- manual
        # LLIR-scheduler-style throttle. k==0 uses intrinsic (zero-init).
        F8_IR_t = fx.Float8E4M3FN.ir_type
        n_blocks = ceildiv(c_n, BLOCK_N)
        lds = fx.SharedAllocator().allocate(SharedStoragePipe).peek()
        a_cur0, a_cur1 = lds.A_lds_cur_0, lds.A_lds_cur_1
        a_nxt0, a_nxt1 = lds.A_lds_next_0, lds.A_lds_next_1
        b_cur0, b_cur1 = lds.B_lds_cur_0, lds.B_lds_cur_1
        b_nxt0, b_nxt1 = lds.B_lds_next_0, lds.B_lds_next_1

        lane_id = fx.thread_idx.x % 64
        wave_id = fx.thread_idx.x // 64
        wave_i = wave_id // 2
        wave_j = wave_id % 2
        tile_i, tile_j = divmod(fx.block_idx.x, n_blocks)

        gA = make_fp8_buffer_tensor(A, F8_IR_t)
        gB = make_fp8_buffer_tensor(B_T, F8_IR_t)
        a_div = fx.logical_divide(gA, fx.make_layout(1, 1))
        b_div = fx.logical_divide(gB, fx.make_layout(1, 1))
        gl_off_a = fp4_g2s_offsets(lane_id, wave_id, K, N_LDS_STEPS_A, BPR)
        gl_off_b = fp4_g2s_offsets(lane_id, wave_id, K, N_LDS_STEPS_B, BPR)

        mfma = MfmaScaleFp4(N_TILES_A, N_TILES_B, asm=asm_mfma, asm_se=asm_se)
        a_g2s = G2SLoader(a_div, gl_off_a, N_LDS_STEPS_A, F8_IR_t, wave_id)
        b_g2s = G2SLoader(b_div, gl_off_b, N_LDS_STEPS_B, F8_IR_t, wave_id)
        a_s2r = S2RLoaderFp4(wave_i, N_TILES_A, N_SUB, BPR, LDS_ROW_STRIDE, pad=not asm_mfma)
        b_s2r = S2RLoaderFp4(wave_j, N_TILES_B, N_SUB, BPR, LDS_ROW_STRIDE, pad=not asm_mfma)
        sa_s2r = ScaleS2R(A_scale, c_m, K, N_TILES_A)
        sb_s2r = ScaleS2R(B_scale, c_n, K, N_TILES_B)
        store_c = StoreCPlain(C, c_m, c_n, mfma.idx, N_TILES_A, N_TILES_B)

        wave_i_off = wave_i * (N_TILES_A * 16)
        wave_j_off = wave_j * (N_TILES_B * 16)
        sa_b0 = fx.Int32(tile_i * BLOCK_M + wave_i_off)
        sa_b1 = sa_b0 + fx.Int32(LDS_BLOCK_M)
        sb_b0 = fx.Int32(tile_j * BLOCK_N + wave_j_off)
        sb_b1 = sb_b0 + fx.Int32(LDS_BLOCK_N)
        A0o = tile_i * BLOCK_M * K2
        A1o = (tile_i * BLOCK_M + LDS_BLOCK_M) * K2
        B0o = tile_j * BLOCK_N * K2
        B1o = (tile_j * BLOCK_N + LDS_BLOCK_N) * K2

        c00 = [mfma.zero_value] * N_ACCUMS
        c01 = [mfma.zero_value] * N_ACCUMS
        c10 = [mfma.zero_value] * N_ACCUMS
        c11 = [mfma.zero_value] * N_ACCUMS

        # prologue: load k=0 into cur
        a_g2s.load(a_cur0, A0o + 0 * KSTEP)
        a_g2s.load(a_cur1, A1o + 0 * KSTEP)
        b_g2s.load(b_cur0, B0o + 0 * KSTEP)
        b_g2s.load(b_cur1, B1o + 0 * KSTEP)
        wait_barrier(0)

        for k in range_constexpr(K_ITERS - 1):
            a0 = a_s2r.load(a_cur0)
            a1 = a_s2r.load(a_cur1)
            b0 = b_s2r.load(b_cur0)
            b1 = b_s2r.load(b_cur1)
            sa0 = sa_s2r.load(sa_b0, k)
            sa1 = sa_s2r.load(sa_b1, k)
            sb0 = sb_s2r.load(sb_b0, k)
            sb1 = sb_s2r.load(sb_b1, k)
            nk = k + 1
            prefetch = []
            for st in range_constexpr(N_LDS_STEPS_A):
                prefetch.append((a_g2s, a_nxt0, A0o + nk * KSTEP, st))
                prefetch.append((a_g2s, a_nxt1, A1o + nk * KSTEP, st))
            for st in range_constexpr(N_LDS_STEPS_B):
                prefetch.append((b_g2s, b_nxt0, B0o + nk * KSTEP, st))
                prefetch.append((b_g2s, b_nxt1, B1o + nk * KSTEP, st))
            ua = False if k == 0 else asm_mfma
            quads = [(c00, a0, b0, sa0, sb0), (c01, a0, b1, sa0, sb1),
                     (c10, a1, b0, sa1, sb0), (c11, a1, b1, sa1, sb1)]
            _il_mma(mfma, quads, prefetch, ua, N_TILES_A, N_TILES_B)
            wait_barrier(0)
            a_cur0, a_nxt0 = a_nxt0, a_cur0
            a_cur1, a_nxt1 = a_nxt1, a_cur1
            b_cur0, b_nxt0 = b_nxt0, b_cur0
            b_cur1, b_nxt1 = b_nxt1, b_cur1

        # tail: last K-iter, no prefetch
        kt = K_ITERS - 1
        a0 = a_s2r.load(a_cur0)
        a1 = a_s2r.load(a_cur1)
        b0 = b_s2r.load(b_cur0)
        b1 = b_s2r.load(b_cur1)
        sa0 = sa_s2r.load(sa_b0, kt)
        sa1 = sa_s2r.load(sa_b1, kt)
        sb0 = sb_s2r.load(sb_b0, kt)
        sb1 = sb_s2r.load(sb_b1, kt)
        uat = False if kt == 0 else asm_mfma
        quads = [(c00, a0, b0, sa0, sb0), (c01, a0, b1, sa0, sb1),
                 (c10, a1, b0, sa1, sb0), (c11, a1, b1, sa1, sb1)]
        _il_mma(mfma, quads, [], uat, N_TILES_A, N_TILES_B)

        base_row = tile_i * BLOCK_M + wave_i_off
        base_col = tile_j * BLOCK_N + wave_j_off
        store_c.store(c00, base_row + 0, base_col + 0)
        store_c.store(c01, base_row + 0, base_col + LDS_BLOCK_N)
        store_c.store(c10, base_row + LDS_BLOCK_M, base_col + 0)
        store_c.store(c11, base_row + LDS_BLOCK_M, base_col + LDS_BLOCK_N)

    @flyc.kernel(known_block_size=[256, 1, 1])
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
        a_lds0 = lds.A_lds0
        a_lds1 = lds.A_lds1
        b_lds0 = lds.B_lds0
        b_lds1 = lds.B_lds1

        lane_id = fx.thread_idx.x % 64
        wave_id = fx.thread_idx.x // 64
        wave_i = wave_id // 2
        wave_j = wave_id % 2
        tile_i, tile_j = divmod(fx.block_idx.x, n_blocks)

        gA = make_fp8_buffer_tensor(A, F8_IR_t)
        gB = make_fp8_buffer_tensor(B_T, F8_IR_t)
        a_div = fx.logical_divide(gA, fx.make_layout(1, 1))
        b_div = fx.logical_divide(gB, fx.make_layout(1, 1))

        mfma = MfmaScaleFp4(N_TILES_A, N_TILES_B, asm=asm_mfma)
        if const_expr(padded):
            a_g2s = PaddedG2SLoader(A, c_m, K, lane_id, wave_id, N_LDS_STEPS_A, BPR, LDS_ROW_STRIDE)
            b_g2s = PaddedG2SLoader(B_T, c_n, K, lane_id, wave_id, N_LDS_STEPS_B, BPR, LDS_ROW_STRIDE)
        else:
            gl_off_a = fp4_g2s_offsets(lane_id, wave_id, K, N_LDS_STEPS_A, BPR)
            gl_off_b = fp4_g2s_offsets(lane_id, wave_id, K, N_LDS_STEPS_B, BPR)
            a_g2s = G2SLoader(a_div, gl_off_a, N_LDS_STEPS_A, F8_IR_t, wave_id)
            b_g2s = G2SLoader(b_div, gl_off_b, N_LDS_STEPS_B, F8_IR_t, wave_id)
        a_s2r = S2RLoaderFp4(wave_i, N_TILES_A, N_SUB, BPR, LDS_ROW_STRIDE, pad=not asm_mfma)
        b_s2r = S2RLoaderFp4(wave_j, N_TILES_B, N_SUB, BPR, LDS_ROW_STRIDE, pad=not asm_mfma)
        sa_s2r = ScaleS2R(A_scale, c_m, K, N_TILES_A)
        sb_s2r = ScaleS2R(B_scale, c_n, K, N_TILES_B)
        store_c = StoreCPlain(C, c_m, c_n, mfma.idx, N_TILES_A, N_TILES_B)

        wave_i_off = wave_i * (N_TILES_A * 16)
        wave_j_off = wave_j * (N_TILES_B * 16)
        sa_base0 = fx.Int32(tile_i * BLOCK_M + wave_i_off)
        sa_base1 = sa_base0 + fx.Int32(LDS_BLOCK_M)
        sb_base0 = fx.Int32(tile_j * BLOCK_N + wave_j_off)
        sb_base1 = sb_base0 + fx.Int32(LDS_BLOCK_N)

        A0_off = tile_i * BLOCK_M * K2
        A1_off = (tile_i * BLOCK_M + LDS_BLOCK_M) * K2
        B0_off = tile_j * BLOCK_N * K2
        B1_off = (tile_j * BLOCK_N + LDS_BLOCK_N) * K2

        c00 = [mfma.zero_value] * N_ACCUMS
        c01 = [mfma.zero_value] * N_ACCUMS
        c10 = [mfma.zero_value] * N_ACCUMS
        c11 = [mfma.zero_value] * N_ACCUMS

        for k in range_constexpr(K_ITERS):
            a_g2s.load(a_lds0, A0_off + k * KSTEP)
            a_g2s.load(a_lds1, A1_off + k * KSTEP)
            b_g2s.load(b_lds0, B0_off + k * KSTEP)
            b_g2s.load(b_lds1, B1_off + k * KSTEP)
            if const_expr(padded):
                wait_barrier_lgkm()
            else:
                wait_barrier(0)

            a0 = a_s2r.load(a_lds0)
            a1 = a_s2r.load(a_lds1)
            b0 = b_s2r.load(b_lds0)
            b1 = b_s2r.load(b_lds1)
            sa0 = [sa_s2r.load(sa_base0, k)]
            sa1 = [sa_s2r.load(sa_base1, k)]
            sb0 = [sb_s2r.load(sb_base0, k)]
            sb1 = [sb_s2r.load(sb_base1, k)]

            # k==0 uses the intrinsic (correct per-accumulator zero-init into
            # distinct AGPRs); k>0 uses inline-asm MFMA (ties to the real AGPR
            # accumulator, no zero needed) so LLVM can't re-cluster the MFMAs.
            ua = False if k == 0 else asm_mfma
            c00 = mfma.call_subs(a0, b0, c00, sa0, sb0, N_SUB, ua)
            c01 = mfma.call_subs(a0, b1, c01, sa0, sb1, N_SUB, ua)
            c10 = mfma.call_subs(a1, b0, c10, sa1, sb0, N_SUB, ua)
            c11 = mfma.call_subs(a1, b1, c11, sa1, sb1, N_SUB, ua)
            rocdl.s_barrier()

        base_row = tile_i * BLOCK_M + wave_i_off
        base_col = tile_j * BLOCK_N + wave_j_off
        store_c.store(c00, base_row + 0, base_col + 0)
        store_c.store(c01, base_row + 0, base_col + LDS_BLOCK_N)
        store_c.store(c10, base_row + LDS_BLOCK_M, base_col + 0)
        store_c.store(c11, base_row + LDS_BLOCK_M, base_col + LDS_BLOCK_N)

    @flyc.kernel(known_block_size=[256, 1, 1])
    def kernel_gemm_pipe(
        A: fx.Tensor,
        B_T: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        c_m: fx.Int32,
        c_n: fx.Int32,
    ):
        # 4-warp (2x2) port of the 8-wave `pipe`: native MFMA + s_setprio bursts +
        # double-buffered cur/next staging + interleaved barriers + 1-deep scale
        # prefetch. 1 wave/SIMD (no cross-wave-per-SIMD barrier tax) + BLOCK_K=256
        # (N_SUB=2 MFMA sub-blocks/iter) mirrors the competitor a4w4.
        F8_IR_t = fx.Float8E4M3FN.ir_type
        n_blocks = ceildiv(c_n, BLOCK_N)
        lds = fx.SharedAllocator().allocate(SharedStoragePipe).peek()
        a_cur0, a_cur1 = lds.A_lds_cur_0, lds.A_lds_cur_1
        a_next0, a_next1 = lds.A_lds_next_0, lds.A_lds_next_1
        b_cur0, b_cur1 = lds.B_lds_cur_0, lds.B_lds_cur_1
        b_next0, b_next1 = lds.B_lds_next_0, lds.B_lds_next_1

        lane_id = fx.thread_idx.x % 64
        wave_id = fx.thread_idx.x // 64
        wave_i = wave_id // 2
        wave_j = wave_id % 2
        tile_i, tile_j = divmod(fx.block_idx.x, n_blocks)

        A0_off = tile_i * BLOCK_M * K2
        A1_off = (tile_i * BLOCK_M + LDS_BLOCK_M) * K2
        B0_off = tile_j * BLOCK_N * K2
        B1_off = (tile_j * BLOCK_N + LDS_BLOCK_N) * K2

        gA = make_fp8_buffer_tensor(A, F8_IR_t)
        gB = make_fp8_buffer_tensor(B_T, F8_IR_t)
        a_div = fx.logical_divide(gA, fx.make_layout(1, 1))
        b_div = fx.logical_divide(gB, fx.make_layout(1, 1))

        mfma = MfmaScaleFp4(N_TILES_A, N_TILES_B, asm=asm_mfma, asm_se=asm_se)
        gl_off_a = fp4_g2s_offsets(lane_id, wave_id, K, N_LDS_STEPS_A, BPR)
        gl_off_b = fp4_g2s_offsets(lane_id, wave_id, K, N_LDS_STEPS_B, BPR)
        a_g2s = G2SLoader(a_div, gl_off_a, N_LDS_STEPS_A, F8_IR_t, wave_id)
        b_g2s = G2SLoader(b_div, gl_off_b, N_LDS_STEPS_B, F8_IR_t, wave_id)
        a_s2r = S2RLoaderFp4(wave_i, N_TILES_A, N_SUB, BPR, LDS_ROW_STRIDE, pad=not asm_mfma)
        b_s2r = S2RLoaderFp4(wave_j, N_TILES_B, N_SUB, BPR, LDS_ROW_STRIDE, pad=not asm_mfma)
        sa_s2r = ScaleS2R(A_scale, c_m, K, N_TILES_A)
        sb_s2r = ScaleS2R(B_scale, c_n, K, N_TILES_B)
        store_c = StoreCPlain(C, c_m, c_n, mfma.idx, N_TILES_A, N_TILES_B)

        wave_i_off = wave_i * (N_TILES_A * 16)
        wave_j_off = wave_j * (N_TILES_B * 16)
        sa_b0 = fx.Int32(tile_i * BLOCK_M + wave_i_off)
        sa_b1 = sa_b0 + fx.Int32(LDS_BLOCK_M)
        sb_b0 = fx.Int32(tile_j * BLOCK_N + wave_j_off)
        sb_b1 = sb_b0 + fx.Int32(LDS_BLOCK_N)

        def _sa(base, kiter):
            return [sa_s2r.load(base, N_SUB * kiter + s) for s in range_constexpr(N_SUB)]

        def _sb(base, kiter):
            return [sb_s2r.load(base, N_SUB * kiter + s) for s in range_constexpr(N_SUB)]

        c00_frag = [mfma.zero_value] * N_ACCUMS
        c01_frag = [mfma.zero_value] * N_ACCUMS
        c10_frag = [mfma.zero_value] * N_ACCUMS
        c11_frag = [mfma.zero_value] * N_ACCUMS

        b_g2s.load(b_cur0, B0_off + 0 * KSTEP)
        a_g2s.load(a_cur0, A0_off + 0 * KSTEP)
        b_g2s.load(b_cur1, B1_off + 0 * KSTEP)
        a_g2s.load(a_cur1, A1_off + 0 * KSTEP)
        wait_barrier(N_LDS_STEPS_A + N_LDS_STEPS_B)

        b_g2s.load(b_next0, B0_off + 1 * KSTEP)
        a_g2s.load(a_next0, A0_off + 1 * KSTEP)
        b_g2s.load(b_next1, B1_off + 1 * KSTEP)
        wait_barrier(N_LDS_STEPS_A + 2 * N_LDS_STEPS_B)

        sa0 = _sa(sa_b0, 0)
        sa1 = _sa(sa_b1, 0)
        sb0 = _sb(sb_b0, 0)
        sb1 = _sb(sb_b1, 0)

        for k in range_constexpr(K_ITERS - 2):
            ua = False if k == 0 else asm_mfma
            sa0n = _sa(sa_b0, k + 1)

            b0_frag = b_s2r.load(b_cur0)
            a0_frag = a_s2r.load(a_cur0)
            a_g2s.load(a_next1, A1_off + (k + 1) * KSTEP)
            rocdl.s_barrier()
            rocdl.s_setprio(1)
            c00_frag = mfma.call_subs(a0_frag, b0_frag, c00_frag, sa0, sb0, N_SUB, ua)
            rocdl.s_setprio(0)
            rocdl.s_barrier()

            b1_frag = b_s2r.load(b_cur1)
            b_g2s.load(b_cur0, B0_off + (k + 2) * KSTEP)
            sb0n = _sb(sb_b0, k + 1)
            sb1n = _sb(sb_b1, k + 1)
            rocdl.s_barrier()
            rocdl.s_setprio(1)
            c01_frag = mfma.call_subs(a0_frag, b1_frag, c01_frag, sa0, sb1, N_SUB, ua)
            rocdl.s_setprio(0)
            rocdl.s_barrier()

            a1_frag = a_s2r.load(a_cur1)
            a_g2s.load(a_cur0, A0_off + (k + 2) * KSTEP)
            sa1n = _sa(sa_b1, k + 1)
            rocdl.s_barrier()
            rocdl.s_setprio(1)
            c10_frag = mfma.call_subs(a1_frag, b0_frag, c10_frag, sa1, sb0, N_SUB, ua)
            rocdl.s_setprio(0)
            rocdl.s_barrier()

            b_g2s.load(b_cur1, B1_off + (k + 2) * KSTEP)
            wait_barrier(2 * N_LDS_STEPS_A + N_LDS_STEPS_B)
            rocdl.s_setprio(1)
            c11_frag = mfma.call_subs(a1_frag, b1_frag, c11_frag, sa1, sb1, N_SUB, ua)
            rocdl.s_setprio(0)
            rocdl.s_barrier()

            a_cur0, a_next0 = a_next0, a_cur0
            a_cur1, a_next1 = a_next1, a_cur1
            b_cur0, b_next0 = b_next0, b_cur0
            b_cur1, b_next1 = b_next1, b_cur1
            sa0, sa1 = sa0n, sa1n
            sb0, sb1 = sb0n, sb1n

        # Step k = K_ITERS - 2 (prefetch last iter into next)
        k = K_ITERS - 2
        sa0n = _sa(sa_b0, K_ITERS - 1)
        sa1n = _sa(sa_b1, K_ITERS - 1)
        sb0n = _sb(sb_b0, K_ITERS - 1)
        sb1n = _sb(sb_b1, K_ITERS - 1)

        b0_frag = b_s2r.load(b_cur0)
        a0_frag = a_s2r.load(a_cur0)
        rocdl.s_barrier()
        rocdl.s_setprio(1)
        c00_frag = mfma.call_subs(a0_frag, b0_frag, c00_frag, sa0, sb0, N_SUB, asm_mfma)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        b1_frag = b_s2r.load(b_cur1)
        rocdl.s_barrier()
        rocdl.s_setprio(1)
        c01_frag = mfma.call_subs(a0_frag, b1_frag, c01_frag, sa0, sb1, N_SUB, asm_mfma)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        a1_frag = a_s2r.load(a_cur1)
        a_g2s.load(a_next1, A1_off + (K_ITERS - 1) * KSTEP)
        rocdl.s_barrier()
        rocdl.s_setprio(1)
        c10_frag = mfma.call_subs(a1_frag, b0_frag, c10_frag, sa1, sb0, N_SUB, asm_mfma)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        b0_frag = b_s2r.load(b_next0)
        rocdl.s_barrier()
        rocdl.s_setprio(1)
        c11_frag = mfma.call_subs(a1_frag, b1_frag, c11_frag, sa1, sb1, N_SUB, asm_mfma)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        a_cur0, a_next0 = a_next0, a_cur0
        a_cur1, a_next1 = a_next1, a_cur1
        b_cur0, b_next0 = b_next0, b_cur0
        b_cur1, b_next1 = b_next1, b_cur1
        sa0, sa1 = sa0n, sa1n
        sb0, sb1 = sb0n, sb1n

        # Step k = K_ITERS - 1 (drain)
        a0_frag = a_s2r.load(a_cur0)
        a1_frag = a_s2r.load(a_cur1)
        b1_frag = b_s2r.load(b_cur1)
        wait_barrier(0)
        rocdl.s_setprio(1)
        c00_frag = mfma.call_subs(a0_frag, b0_frag, c00_frag, sa0, sb0, N_SUB, asm_mfma)
        c01_frag = mfma.call_subs(a0_frag, b1_frag, c01_frag, sa0, sb1, N_SUB, asm_mfma)
        c10_frag = mfma.call_subs(a1_frag, b0_frag, c10_frag, sa1, sb0, N_SUB, asm_mfma)
        c11_frag = mfma.call_subs(a1_frag, b1_frag, c11_frag, sa1, sb1, N_SUB, asm_mfma)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        base_row = tile_i * BLOCK_M + wave_i_off
        base_col = tile_j * BLOCK_N + wave_j_off
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
        kern = kernel_gemm_pipe if mode == "pipe" else (kernel_gemm_il if interleave else kernel_gemm)
        kern(
            A, B_T, C, A_scale, B_scale, c_m, c_n,
            value_attrs={"rocdl.waves_per_eu": 1, "rocdl.flat_work_group_size": "256,256"},
        ).launch(grid=(grid_x, 1, 1), block=(256, 1, 1), stream=stream)

    return launch_gemm
