# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""MXFP4 dense GEMM — 4-wave (2x2) port aiming to unlock AGPR-accumulator + N-slice
(the gluon a4w4 recipe) that the 8-wave production (turbo/mxfp4_gemm_8wave.py) cannot
express (8-wave -> 2 waves/SIMD -> only 128 AGPR/wave -> forced AGPR spills).

4-wave => 1 wave/SIMD => the full 256-AGPR file is available for one wave, so the
256-f32 accumulator (acc_left + acc_right, N-sliced) lives cleanly in AGPR and frees
arch VGPR for operand/fragment prefetch.

CORRECTNESS-FIRST v0: single-LDS-buffer, full s_barrier per K-iter, broadcast scale.
Validates the 2x2 tiling + N-slice + AGPR path. Pipeline/scale-pack optimization
follows once correct.

Topology: 4 waves, wave_m = wave_id // 2 in {0,1}, wave_n = wave_id % 2 in {0,1}.
Tile BM=BN=256, BK=256 (N_SUB=2). Each wave: M = wave_m*128 (8 16-tiles), N split
into left/right 128-halves; within a half wave_n*64 (4 16-tiles). acc_left/acc_right
each = N_TILES_A(8) x N_TILES_B_SLICE(4) = 32 accums.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import buffer_ops, const_expr, range_constexpr, rocdl, primitive
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from kernels.fp8_gemm_utils import G2SLoader, ceildiv, make_fp8_buffer_tensor, wait_barrier
from turbo.mxfp8_gemm_8wave import ScaleS2R, StoreCPlain, preshuffle_scale
from turbo.mxfp4_gemm_8wave import (
    MfmaScaleFp4,
    S2RLoaderFp4,
    ScaleS2RPacked,
    ScaleS2RPackedA2,
    fp4_g2s_offsets,
    grouped_xcd_pid,
    preshuffle_scale_packed,
    preshuffle_scale_packed_a2,
)


# Production 4-wave config: bare-asm whole-loop INPLACE-DIAG + SCVGPR (scales direct to VGPR,
# no LDS ds_read) with FP4_INPLACE_1BAR=0 (per-phase s_barrier — the cross-wave LDS sync that
# makes SCVGPR det0; with 1BAR=1 the shortened stream races). = ~5350 TF det0 vs 5198 INPLACE-only.
# Applied via setdefault so any explicit env var still overrides (experiments). FP4_PROD=0 disables.
_PROD_DEFAULTS = {
    "FP4_ASMMFMA": "6", "FP4_INPLACE": "1", "FP4_INPLACE_DIAG": "1", "FP4_MMORD": "3",
    "FP4_SINNER": "1", "FP4_INPLACE_1BAR": "0", "FP4_INPLACE_ELGK": "9", "FP4_WLVMCN": "10",
    "FP4_MMORD": "5",           # 2x4 wider N-block order: better B-operand reuse
    "FP4_INPLACE_ALT": "0",     # B-side progressive (complements MMORD=5)
    "FP4_INPLACE_GAVOID": "1",  # avoid g2s in refill-free slots -> better LDS bandwidth
    "FP4_WLBARNOP": "1",       # 1 s_nop after barrier: settle time for barrier -> smoother ds_read start
    "FP4_SC_VGPR": "1", "FP4_PIN": "1", "FP4_PINSC": "1", "FP4_PINBASE": "8",
    "FP4_SCV_ILV": "1",   # interleave scale buffer_load into mfma stream (overlaps mfma, frees boundary vmem slot)
}


def _apply_prod_defaults():
    import os as _osp
    if int(_osp.environ.get("FP4_PROD", "1")):
        for _k, _v in _PROD_DEFAULTS.items():
            _osp.environ.setdefault(_k, _v)


def preshuffle_mxfp4_scales_4w(a_e8m0, b_e8m0, K, BLOCK_M=256, BLOCK_N=256, packed=True):
    """Host scale prep for the 4-wave kernel.

    packed=True (scale-shuffle, gluon-style): pack each wave's 4 sub-tile E8M0 into
    the 4 bytes of ONE i32 (byte t = tile t), MFMA selects via opsel. A's 128-M-row
    (8 tiles) wave coverage = 2 packed groups (64 rows each) -> kernel loads 2 dwords;
    B's 64-N-col slice (4 tiles) = 1 group -> 1 dword. 4x less scale VMEM than the
    broadcast-dwordx4 path below.

    packed=False (legacy broadcast): A laid out n_tiles=4 (kernel issues two dwordx4
    loads, rows 0-63 + 64-127); B n_tiles=4 (wave_n 64 N-cols/slice).
    """
    import os as _os9
    _apply_prod_defaults()
    if packed:
        if int(_os9.environ.get("FP4_SC_VGPR", "0")) or int(_os9.environ.get("FP4_SCDWX4", "0")):
            # lane-contiguous so a lane's n_regions*n_sub packed dwords are contiguous -> ONE
            # buffer_load_dwordx4 per operand. SC_VGPR: to VGPR (no LDS). SCDWX4: to LDS (det-safe).
            # n_regions=2 (A:g0,g1 / B:BL,BR). n_sub from block_k (default BK256 -> 2).
            from .mxfp4_gemm_8wave import preshuffle_scale_lane_contig
            _ns = max(int(_os9.environ.get("BLOCK_K", "256")) // 128, 1)
            return (preshuffle_scale_lane_contig(a_e8m0, K, 4, _ns, 'A'),
                    preshuffle_scale_lane_contig(b_e8m0, K, 4, _ns, 'B'))
        if int(_os9.environ.get("FP4_SC_A2", "0")):
            # A combined dwordx2 (g0,g1 coalesced) -> 1 scale load for A instead of 2.
            return preshuffle_scale_packed_a2(a_e8m0, K, 4), preshuffle_scale_packed(b_e8m0, K, 4)
        return preshuffle_scale_packed(a_e8m0, K, 4), preshuffle_scale_packed(b_e8m0, K, 4)
    return preshuffle_scale(a_e8m0, K, 4), preshuffle_scale(b_e8m0, K, 4)


def compile_mxfp4_gemm_4w(
    *,
    K: int,
    BLOCK_M: int = 256,
    BLOCK_N: int = 256,
    block_k: int = 256,
    group_m: int = 4,
    num_xcds: int = 8,
    group_n: int = 16,   # 2D band L2 lever: +3.8% on wide-N (gate/up), neutral on N=8192.
    swizzle: bool = True,
    agpr: bool = True,
    pad: bool = True,
    wait: int = 0,
    packed_scale: bool = True,
    maxnreg: int = 0,
):
    PAD = pad
    WAIT = wait
    PACKED = packed_scale
    BLOCK_K = block_k
    assert BLOCK_M == 256 and BLOCK_N in (128, 256), "4-wave: BM=256, BN in {128,256}"
    assert BLOCK_K % 128 == 0 and K % BLOCK_K == 0
    import os as _os0
    _apply_prod_defaults()
    # BLOCK_N=128: single N-slice (no BR), HALF the accs (32->128 AGPR) + 2-buffer A so
    # V+A<=256 AND LDS<=80K/WG -> 2 waves/SIMD (occ=2). occ=2 hides the ds_read LATENCY
    # (PMC: LdsUtil 8% non-bw-bound, big SQ_WAIT_INST_LDS) that 1-wave/SIMD exposes.
    HAS_BR = const_expr(BLOCK_N == 256)
    _ASMM_E = const_expr(int(_os0.environ.get("FP4_ASMMFMA", "0")))
    # FP4_WLRING: 4-buffer ring whole-loop (hide ds_read via register-prefetch + refill-OTHER
    # 2-ahead + staggered vmcnt; BK128 only, LDS=4xA(64K)+4xBL/BR(64K)+4xSC<160K). unroll-4.
    _RING = const_expr(_ASMM_E == 6 and int(_os0.environ.get("FP4_WLRING", "0")) == 1)
    # FP4_WLRING_2A: TRUE 2-ahead g2s (prefill buf2 k=2, in-loop g2s from k=3) -> deep vmcnt correct.
    _R2A = const_expr(_RING and int(_os0.environ.get("FP4_WLRING_2A", "0")) == 1)
    # FP4_SUBSTREAM: sub-granular streaming, NSS=2 reg sub-sets + N LDS buffers (refill-SAME
    # nbuf-ahead). Fits BK256 (sub-level regs) where NSET=2 tile-level overflows. FP4_SS_NBUF=N (3).
    _SS = const_expr(_ASMM_E == 6 and int(_os0.environ.get("FP4_SUBSTREAM", "0")) == 1)
    _SSNB = const_expr(int(_os0.environ.get("FP4_SS_NBUF", "3")))
    # FP4_SC_A2: A-scale combined dwordx2 (g0,g1 coalesced) -> 1 scale load for A instead of 2.
    _SCA2 = const_expr(_ASMM_E == 6 and PACKED and int(_os0.environ.get("FP4_SC_A2", "0")) == 1)
    # FP4_ASYM: asymmetric 2A+4B sub-granular (A pool shallow, B pool deep -> deep B g2s in-flight).
    _ASYM = const_expr(_ASMM_E == 6 and int(_os0.environ.get("FP4_ASYM", "0")) == 1)
    _ANB = const_expr(int(_os0.environ.get("FP4_ASYM_NA", "2")))   # A pool
    _BNB = const_expr(int(_os0.environ.get("FP4_ASYM_NB", "4")))   # B pool
    _PRELL = const_expr(_SSNB if _SS else (3 if _R2A else 2))   # operand/scale buffers prefilled (k=0..PRELL-1)
    _PRELL_A = const_expr(_ANB if _ASYM else _PRELL)
    _PRELL_B = const_expr(_BNB if _ASYM else _PRELL)
    # ASMMFMA=6 whole-loop bare-asm: 2 A/B buffers (unroll-2) or 4 (ring) or _SSNB (substream) or asym(_ANB/_BNB).
    NBB = const_expr(_BNB if _ASYM else (4 if _RING else (_SSNB if _SS else 2)))  # B/SC pool
    NABUF = const_expr(_ANB if _ASYM else (4 if _RING else (_SSNB if _SS else (2 if _ASMM_E == 6 else int(_os0.environ.get("FP4_NABUF", "3" if BLOCK_N == 256 else "2"))))))
    OCC = const_expr(int(_os0.environ.get("FP4_OCC", "1" if BLOCK_N == 256 else "2")))

    KI = K // BLOCK_K
    N_SUB = BLOCK_K // 128
    BPR = BLOCK_K // 2          # packed-fp4 bytes per K-iter row in LDS
    KSTEP = BPR
    K2 = K // 2                 # packed-fp4 gmem row stride (bytes)

    # 2x2 wave topology. BN256: B split into L/R halves (128 each), 2 slices. BN128:
    # ONE slice covering the full BLOCK_N=128 (= exactly one BN256 slice) -> 32 accs.
    N_TILES_A = BLOCK_M // 32                          # 8: wave_m covers 128 M-rows
    LDS_BN_HALF = (BLOCK_N // 2) if HAS_BR else BLOCK_N  # slice width (128 for both)
    N_TILES_BH = LDS_BN_HALF // 32                     # 4: wave_n covers 64 N-cols/slice

    # FP4_WLPAD: pad LDS row stride to reduce g2s-write vs ds_read-read bank conflicts (wholeloop).
    _WLPAD = const_expr(int(_os0.environ.get("FP4_WLPAD", "0")))
    LDS_ROW_STRIDE = BPR + _WLPAD
    a_lds_size = BLOCK_M * LDS_ROW_STRIDE        # 256 rows
    bh_lds_size = LDS_BN_HALF * LDS_ROW_STRIDE   # 128 rows per B half

    # G2S step coverage: rows/step = (64/lpr) * n_waves, lpr = BPR//16
    _ROWS_PER_STEP = 64 // (BPR // 16) * (256 // 64)   # n_waves = 256//64 = 4
    N_LDS_STEPS_A = BLOCK_M // _ROWS_PER_STEP
    N_LDS_STEPS_BH = LDS_BN_HALF // _ROWS_PER_STEP

    LPI = N_LDS_STEPS_A + 2 * N_LDS_STEPS_BH  # tile buffer_load_to_lds per K-iter

    # A is NABUF-stage so A[k+NABUF-1] G2S can stay in flight across the MFMA region
    # (vmcnt-stagger) -> hides gmem latency. BN256: 3xA(96K)+2xBL/BR(64K)=160K (1WG/CU,
    # occ=1). BN128: 2xA(64K)+2xBL(16K)=80K -> 2WG/CU (occ=2) hides ds_read latency.
    _anns = {f"A_lds{i}": fx.Array[fx.Float8E4M3FN, a_lds_size, 16] for i in range_constexpr(NABUF)}
    for _b in range_constexpr(NBB):
        _anns[f"BL_lds{_b}"] = fx.Array[fx.Float8E4M3FN, bh_lds_size, 16]
    if const_expr(HAS_BR):
        for _b in range_constexpr(NBB):
            _anns[f"BR_lds{_b}"] = fx.Array[fx.Float8E4M3FN, bh_lds_size, 16]
    if const_expr(_ASMM_E == 6):
        # whole-loop scales-in-LDS: NBB ping-pong buffers, each holds one K-iter's packed
        # scales: n_waves(4) x 4 groups(A-g0,A-g1,BL,BR) x N_SUB subs x 64 lanes (1 dword).
        _SCBUF = 4 * 4 * (BLOCK_K // 128) * 64    # n_waves * groups * n_sub * 64 dwords
        # SCDWX4 2-ahead needs 4 SC_lds buffers (double-buffer per phase) for deep prefetch (aiter pattern):
        # dwordx4-lds scale loaded 2 phases ahead -> drained by vmcnt(WLV) + barrier-between => det0+fast.
        _NSCBUF = const_expr(4 if int(__import__("os").environ.get("FP4_SCDWX4_2A", "0")) else NBB)
        for _b in range_constexpr(_NSCBUF):
            _anns[f"SC_lds{_b}"] = fx.Array[fx.Int32, _SCBUF, 16]
        if const_expr(_SCA2):
            # a2 A-scale LDS: per wave, [sub][lane][g0,g1] interleaved (dwordx2 g2s + b64 ds_read).
            _SCA2BUF = 4 * (BLOCK_K // 128) * 64 * 2   # n_waves * n_sub * 64 * 2(g0,g1)
            for _b in range_constexpr(NBB):
                _anns[f"SCA_lds{_b}"] = fx.Array[fx.Int32, _SCA2BUF, 16]
    SharedStorageFp4_4w = fx.struct(type("SharedStorageFp4_4w", (), {"__annotations__": _anns}))

    @flyc.kernel(known_block_size=[256, 1, 1])
    def kernel_gemm_4w(
        A: fx.Tensor,
        B_T: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        c_m: fx.Int32,
        c_n: fx.Int32,
    ):
        F8_IR_t = fx.Float8E4M3FN.ir_type
        lds = fx.SharedAllocator().allocate(SharedStorageFp4_4w).peek()
        A_buf = [getattr(lds, f"A_lds{i}") for i in range_constexpr(NABUF)]
        BL_buf = [getattr(lds, f"BL_lds{i}") for i in range_constexpr(NBB)]
        BR_buf = [getattr(lds, f"BR_lds{i}") for i in range_constexpr(NBB)] if const_expr(HAS_BR) else BL_buf

        lane_id = fx.thread_idx.x % 64
        wave_id = fx.thread_idx.x // 64
        wave_m = wave_id // 2
        wave_n = wave_id % 2
        block_m, block_n = grouped_xcd_pid(fx.block_idx.x, c_m, c_n, BLOCK_M, BLOCK_N,
                                           group_m=group_m, num_xcds=num_xcds, group_n=group_n)

        A_off = block_m * BLOCK_M * K2
        BL_off = block_n * BLOCK_N * K2
        BR_off = (block_n * BLOCK_N + LDS_BN_HALF) * K2

        gA = make_fp8_buffer_tensor(A, F8_IR_t)
        gB = make_fp8_buffer_tensor(B_T, F8_IR_t)
        a_div = fx.logical_divide(gA, fx.make_layout(1, 1))
        b_div = fx.logical_divide(gB, fx.make_layout(1, 1))

        mfma = MfmaScaleFp4(N_TILES_A, N_TILES_BH, packed=PACKED)

        gl_off_a = fp4_g2s_offsets(lane_id, wave_id, K, N_LDS_STEPS_A, BPR, swizzle=swizzle)
        gl_off_b = fp4_g2s_offsets(lane_id, wave_id, K, N_LDS_STEPS_BH, BPR, swizzle=swizzle)
        # G2S-in-asm (FP4_ASMMFMA=3): raw buffer resources + LDS m0 helper so the
        # next-iter buffer_load_lds can be hand-interleaved INSIDE the MFMA asm cluster.
        _NW = fx.block_dim.x // 64
        rsrc_a = buffer_ops.create_buffer_resource(A, max_size=False, num_records_bytes=c_m * K2)
        rsrc_b = buffer_ops.create_buffer_resource(B_T, max_size=False, num_records_bytes=c_n * K2)
        def _m0(buf, step):
            v = fx.Int32(fx.ptrtoint(buf.ptr)) + fx.Int32(wave_id) * fx.Int32(1024) + fx.Int32(step * _NW * 1024)
            return rocdl.readfirstlane(T.i32, v)  # m0 must be uniform SGPR
        a_g2s = G2SLoader(a_div, gl_off_a, N_LDS_STEPS_A, F8_IR_t, wave_id)
        bl_g2s = G2SLoader(b_div, gl_off_b, N_LDS_STEPS_BH, F8_IR_t, wave_id)
        br_g2s = G2SLoader(b_div, gl_off_b, N_LDS_STEPS_BH, F8_IR_t, wave_id)

        # FP4_ASMMFMA: hand-asm packed MFMA cluster (forces MFMA order for hand-interleave
        # toward gluon-density). Needs i32x4 frags (pad=False) like the asm path.
        _ASMM = const_expr(int(__import__("os").environ.get("FP4_ASMMFMA", "0")))
        _PAD = const_expr(PAD and not _ASMM)
        a_s2r = S2RLoaderFp4(wave_m, N_TILES_A, N_SUB, BPR, LDS_ROW_STRIDE, pad=_PAD, swizzle=swizzle)
        b_s2r = S2RLoaderFp4(wave_n, N_TILES_BH, N_SUB, BPR, LDS_ROW_STRIDE, pad=_PAD, swizzle=swizzle)

        # A scale: 8 M-tiles span 2 x 64-row groups (4 tiles each). packed -> ONE i32
        # per group (byte t = tile t, opsel selects), 4x less VMEM than broadcast-dwordx4.
        if const_expr(PACKED):
            # group_span-ceil the dim (64 = 16*n_tiles) so the floor in the loader's
            # record count covers the edge scale group (matches 8-wave _sa_qd).
            _qm = ((c_m + 63) // 64) * 64
            _qn = ((c_n + 63) // 64) * 64
            sa_s2r = ScaleS2RPackedA2(A_scale, _qm, K, 4) if const_expr(_SCA2) else ScaleS2RPacked(A_scale, _qm, K, 4)
            sb_s2r = ScaleS2RPacked(B_scale, _qn, K, 4)
        else:
            sa_s2r = ScaleS2R(A_scale, c_m, K, 4)
            sb_s2r = ScaleS2R(B_scale, c_n, K, N_TILES_BH)
        store_c = StoreCPlain(C, c_m, c_n, mfma.idx, N_TILES_A, N_TILES_BH)

        def _mfma_packed(a, b, c, sa2, sb1):
            """packed MFMA: sa2[s] = [dword tiles0-3, dword tiles4-7] (opsel = i%4,
            dword = i//4); sb1[s] = ONE dword (4 N-tiles, opsel = j)."""
            for s in range_constexpr(N_SUB):
                for i in range_constexpr(N_TILES_A):
                    for j in range_constexpr(N_TILES_BH):
                        idx = i * N_TILES_BH + j
                        c[idx] = mfma._do_packed(
                            a[i][s], b[j][s], c[idx], sa2[s][i // 4], i % 4, sb1[s], j)
            return c

        wave_m_off = wave_m * (N_TILES_A * 16)        # 0 or 128
        wave_n_off = wave_n * (N_TILES_BH * 16)       # 0 or 64
        sa_base = fx.Int32(block_m * BLOCK_M + wave_m_off)
        sbl_base = fx.Int32(block_n * BLOCK_N + wave_n_off)
        sbr_base = fx.Int32(block_n * BLOCK_N + LDS_BN_HALF + wave_n_off)

        accL = [mfma.zero_value] * (N_TILES_A * N_TILES_BH)
        accR = [mfma.zero_value] * (N_TILES_A * N_TILES_BH)

        # FP4_APREF: A operand-prefetch. A_buf[(k+1)%3] is loaded at iter k-1 and its G2S
        # lands during k-1's long MFMA -> landed by iter k. So we can ds_read NEXT iter's A
        # at iter k (carried), letting its latency overlap THIS iter's MFMA. Hides A's
        # ds_read (16/32 operand frags); B stays in-iter (B is only 1-ahead, loaded THIS
        # iter -> not landed yet). No extra LDS (within the 3-buf A ring).
        _APREF = const_expr(int(__import__("os").environ.get("FP4_APREF", "0")))

        # -- Prologue: A is (NABUF-1)-ahead -> prefill A[0..NABUF-2]; B 1-ahead -> B[0] --
        AHEAD = const_expr(NABUF - 1)
        a_g2s.load(A_buf[0], A_off + 0 * KSTEP)
        bl_g2s.load(BL_buf[0], BL_off + 0 * KSTEP)
        if const_expr(HAS_BR):
            br_g2s.load(BR_buf[0], BR_off + 0 * KSTEP)
        for _p in range_constexpr(1, AHEAD):
            if const_expr(KI > _p):
                a_g2s.load(A_buf[_p], A_off + _p * KSTEP)
        wait_barrier(0)  # A[0..AHEAD-1],B[0] landed + cross-wave visible before loop

        # APREF prologue: read iter-0's A now (A_buf[0], landed above). Carried into loop.
        a_carry = a_s2r.load(A_buf[0]) if const_expr(_APREF) else None
        # ASMMFMA=5 (cross-iter double-buffer): carry iter-0's A frags (read now, landed).
        a_cur_pf = a_s2r.load(A_buf[0]) if const_expr(_ASMM == 5) else None

        if const_expr(_ASMM == 6):
            # WHOLE-LOOP bare-asm: entire K-loop in ONE asm hw-loop. REAL scales via
            # scales-in-LDS (ds_read_b32, lgkmcnt; not buffer_load-to-VGPR which entangles
            # the g2s vmcnt). 2 LDS buffers ping-pong (ops + scales). unroll-2.
            K128 = const_expr(K // 128)
            _NSCBUF = const_expr(4 if int(__import__("os").environ.get("FP4_SCDWX4_2A", "0")) else NBB)
            SC_buf = [getattr(lds, f"SC_lds{b}") for b in range_constexpr(_NSCBUF)]
            _SCW = const_expr(4 * N_SUB * 64)   # dwords per wave-region per buffer
            # prologue: A pool buf1..PRELL_A-1, B pool buf1..PRELL_B-1 (asym: A shallow, B deep)
            for _pp in range_constexpr(1, _PRELL_A):
                if const_expr(KI > _pp):
                    a_g2s.load(A_buf[_pp], A_off + _pp * KSTEP)
            for _pp in range_constexpr(1, _PRELL_B):
                if const_expr(KI > _pp):
                    bl_g2s.load(BL_buf[_pp], BL_off + _pp * KSTEP)
                    br_g2s.load(BR_buf[_pp], BR_off + _pp * KSTEP)

            _TRB8 = const_expr(int(__import__("os").environ.get("FP4_TRB8", "0")))
            _SC_VGPR = const_expr(int(__import__("os").environ.get("FP4_SC_VGPR", "0")))
            _SCDWX4 = const_expr(int(__import__("os").environ.get("FP4_SCDWX4", "0")))
            _NSCT = const_expr(4 * N_SUB)
            def _sc_store(buf, slot, val):
                # TRB8: lane-major (lane*nsct + slot); else slot-major (slot*64 + lane)
                if const_expr(_TRB8):
                    idx = fx.Int32(wave_id) * fx.Int32(_SCW) + lane_id * fx.Int32(_NSCT) + fx.Int32(slot)
                else:
                    idx = fx.Int32(wave_id) * fx.Int32(_SCW) + fx.Int32(slot * 64) + lane_id
                pp = fx.add_offset(SC_buf[buf].ptr, fx.make_int_tuple(idx))
                primitive.ptr_store(val, pp)
            def _sc_store_a2A(buf, s, region, val):
                # a2 A LDS layout: byte = s*512 + lane*8 + region*4 -> dword idx s*128+lane*2+region
                idx = (fx.Int32(wave_id) * fx.Int32(_SCW) + fx.Int32(s * 128)
                       + lane_id * fx.Int32(2) + fx.Int32(region))
                pp = fx.add_offset(SC_buf[buf].ptr, fx.make_int_tuple(idx))
                primitive.ptr_store(val, pp)
            # SCVGPR: direct-to-VGPR in loop; SCDWX4: lane-packed g2s in loop+prologue -> no host prefill.
            for _bk in range_constexpr(0 if (_SC_VGPR or _SCDWX4) else _PRELL_B):
                for s in range_constexpr(N_SUB):
                    _k128 = N_SUB * _bk + s
                    if const_expr(_SCA2):
                        _va = sa_s2r.load(sa_base, _k128)        # [region0, region1]
                        _sc_store_a2A(_bk, s, 0, _va[0])
                        _sc_store_a2A(_bk, s, 1, _va[1])
                    else:
                        _sc_store(_bk, 0 * N_SUB + s, sa_s2r.load(sa_base, _k128))
                        _sc_store(_bk, 1 * N_SUB + s, sa_s2r.load(sa_base + 64, _k128))
                    _sc_store(_bk, 2 * N_SUB + s, sb_s2r.load(sbl_base, _k128))
                    _sc_store(_bk, 3 * N_SUB + s, sb_s2r.load(sbr_base, _k128))
            # scale stores are ds_write (lgkmcnt); wait_barrier(0) only drains vmcnt ->
            # must drain lgkmcnt too before the asm reads SC_lds (else race).
            _llvm.inline_asm(res=None, operands_=[], asm_string="s_waitcnt lgkmcnt(0)",
                             constraints="", has_side_effects=True)
            wait_barrier(0)

            a_base6 = [[a_s2r.base_addr(A_buf[b], s) for s in range_constexpr(N_SUB)] for b in range_constexpr(NABUF)]
            bl_base6 = [[b_s2r.base_addr(BL_buf[b], s) for s in range_constexpr(N_SUB)] for b in range_constexpr(NBB)]
            br_base6 = [[b_s2r.base_addr(BR_buf[b], s) for s in range_constexpr(N_SUB)] for b in range_constexpr(NBB)]

            def _gbase(buf):
                v = fx.Int32(fx.ptrtoint(buf.ptr)) + fx.Int32(wave_id) * fx.Int32(1024)
                return rocdl.readfirstlane(T.i32, v)
            abase6 = [_gbase(A_buf[b]) for b in range_constexpr(NABUF)]
            blbase6 = [_gbase(BL_buf[b]) for b in range_constexpr(NBB)]
            brbase6 = [_gbase(BR_buf[b]) for b in range_constexpr(NBB)]
            gl_a6 = [fx.Int32(gl_off_a[st]) for st in range_constexpr(N_LDS_STEPS_A)]
            gl_b6 = [fx.Int32(gl_off_b[st]) for st in range_constexpr(N_LDS_STEPS_BH)]
            scv6 = fx.Int32(0x7f7f7f7f)
            soff6_a = rocdl.readfirstlane(T.i32, A_off + fx.Int32(_PRELL_A * KSTEP))
            soff6_bl = rocdl.readfirstlane(T.i32, BL_off + fx.Int32(_PRELL_B * KSTEP))
            soff6_br = rocdl.readfirstlane(T.i32, BR_off + fx.Int32(_PRELL_B * KSTEP))
            # scale LDS read/g2s bases (per-wave region), scale rsrc, voffset, soffset inits
            # TRB8: lane-major LDS (each lane's nsct=4*N_SUB scale dwords contiguous) so they
            # can be read back as ds_read_b64 pairs. read base += lane*nsct; g2s voffset = lane*nsct*4.
            _TRB8 = const_expr(int(__import__("os").environ.get("FP4_TRB8", "0")))
            _NSCT = const_expr(4 * N_SUB)
            _scrb_lane = (lane_id * fx.Int32(_NSCT)) if _TRB8 else ((lane_id * fx.Int32(2 * N_SUB)) if _SCDWX4 else lane_id)  # SCDWX4 lane-packed
            sc_rb6 = [fx.ptrtoint(fx.add_offset(SC_buf[b].ptr,
                      fx.make_int_tuple(fx.Int32(wave_id) * fx.Int32(_SCW) + _scrb_lane)))
                      for b in range_constexpr(_NSCBUF)]
            sc_gb6 = [rocdl.readfirstlane(T.i32, fx.Int32(fx.ptrtoint(fx.add_offset(SC_buf[b].ptr,
                      fx.make_int_tuple(fx.Int32(wave_id) * fx.Int32(_SCW))))))
                      for b in range_constexpr(_NSCBUF)]
            sc_voff6 = lane_id * fx.Int32(4)   # gmem per-lane scale offset (TRB8 uses ds_write for LDS layout)
            def _scsoff(base, extra):
                grp = (base + fx.Int32(extra)) // fx.Int32(64)
                return rocdl.readfirstlane(T.i32, (grp * fx.Int32(K128) + fx.Int32(_PRELL_B * N_SUB)) * fx.Int32(256))
            sc_soff06 = [_scsoff(sa_base, 0), _scsoff(sa_base, 64), _scsoff(sbl_base, 0), _scsoff(sbr_base, 0)]
            # a2 (FP4_SC_A2): A scales as coalesced dwordx2 (g0,g1) -> read base lane*2,
            # voffset lane*8, soffset (wi*K128+k)*512. wi matches ScaleS2RPackedA2.load.
            sca_rb6 = sca_gb6 = sca_voff6 = None
            if const_expr(_SCA2):
                _wi = sa_base // fx.Int32(128)   # g0//2, g0 = sa_base//64 (even)
                sc_soff06 = [rocdl.readfirstlane(T.i32,
                              (_wi * fx.Int32(K128) + fx.Int32(_PRELL_B * N_SUB)) * fx.Int32(512))] + sc_soff06[1:]
                sca_rb6 = [fx.ptrtoint(fx.add_offset(SC_buf[b].ptr,
                           fx.make_int_tuple(fx.Int32(wave_id) * fx.Int32(_SCW) + lane_id * fx.Int32(2))))
                           for b in range_constexpr(NBB)]
                sca_gb6 = sc_gb6
                sca_voff6 = lane_id * fx.Int32(8)
            # FP4_SC_VGPR: lane-contig scale direct-read. rsrc covers full preshuffled tensor;
            # voffset = lane*16 (2*N_SUB dwords/lane); soffset = wi*K128*512 (kk stride 1024).
            # sa_s2r.rsrc is max_size=False sized to the scale tensor (same total bytes as the
            # lane-contig layout) -> last (unused) prefetch OOB clamps to 0 (deterministic).
            _scrsa_v = sa_s2r.rsrc; _scrsb_v = sb_s2r.rsrc
            if const_expr(_SC_VGPR or _SCDWX4):
                _wia = sa_base // fx.Int32(128)
                _wib = (sbl_base // fx.Int32(256)) * fx.Int32(2) + (sbl_base % fx.Int32(256)) // fx.Int32(64)
                _soa = rocdl.readfirstlane(T.i32, _wia * fx.Int32(K128) * fx.Int32(512))
                _sob = rocdl.readfirstlane(T.i32, _wib * fx.Int32(K128) * fx.Int32(512))
                sc_soff06 = [_soa, sc_soff06[1], _sob, sc_soff06[3]]
                sc_voff6 = lane_id * fx.Int32(8 * N_SUB)   # 2*N_SUB dwords/lane * 4B (was hardcoded 16 for N_SUB=2)
            accL, accR = mfma.call_mxfp4_wholeloop(
                a_base6, bl_base6, br_base6, a_s2r.tile_stride, b_s2r.tile_stride,
                abase6, blbase6, brbase6, gl_a6, gl_b6, rsrc_a, rsrc_b,
                fx.Int32(KSTEP), scv6, accL, accR, N_SUB, N_LDS_STEPS_A, N_LDS_STEPS_BH,
                fx.Int32(KI), soff6_a, soff6_bl, soff6_br,
                sc_rb6, sc_gb6, _scrsa_v, _scrsb_v, sc_voff6, sc_soff06,
                sca_rb6, sca_gb6, sca_voff6)

        # FP4_LEANBAR: speed-ceiling probe. 2 = strip ALL sync (s_barrier + vmcnt drain,
        # correctness ignored) -> measures the no-sync ceiling; 1 = drop only end s_barrier.
        _LEAN = const_expr(int(__import__("os").environ.get("FP4_LEANBAR", "0")))
        for k in range_constexpr(0 if _ASMM == 6 else KI):
            cur_a = k % NABUF
            cur_b = k % 2
            nxt_b = (k + 1) % 2
            wr_a = (k + AHEAD) % NABUF          # A (NABUF-1)-ahead (WAR-free: last read iter k-1)
            has_b = k + 1 < KI
            has_a = k + AHEAD < KI

            # Scales (buffer_load->VGPR; vmcnt is handled by the MFMA's data dependency,
            # NOT on the explicit LDS drain path).
            if const_expr(PACKED):
                sa = [[sa_s2r.load(sa_base, N_SUB * k + s), sa_s2r.load(sa_base + 64, N_SUB * k + s)]
                      for s in range_constexpr(N_SUB)]
                sbl = [sb_s2r.load(sbl_base, N_SUB * k + s) for s in range_constexpr(N_SUB)]
                sbr = [sb_s2r.load(sbr_base, N_SUB * k + s) for s in range_constexpr(N_SUB)] if const_expr(HAS_BR) else None
            else:
                sa = [sa_s2r.load(sa_base, N_SUB * k + s) + sa_s2r.load(sa_base + 64, N_SUB * k + s)
                      for s in range_constexpr(N_SUB)]
                sbl = [sb_s2r.load(sbl_base, N_SUB * k + s) for s in range_constexpr(N_SUB)]
                sbr = [sb_s2r.load(sbr_base, N_SUB * k + s) for s in range_constexpr(N_SUB)] if const_expr(HAS_BR) else None

            # READS FIRST: A[cur](2-ahead) + B[cur](1-ahead) are landed + cross-wave visible
            # from the PREVIOUS iter's end wait_barrier -> no stall (key vs old "drain just-
            # issued B before reads" which exposed the full B latency).
            # FP4_NODSR: speed probe — skip operand ds_read (constant frags) to isolate
            # whether ds_read is the ceiling bottleneck (vs MFMA+G2S). Garbage output.
            if const_expr(_ASMM in (4, 5)):
                a_frag = bl_frag = br_frag = None   # the asm does its own ds_read
            elif const_expr(int(__import__("os").environ.get("FP4_NODSR", "0"))):
                _cf4 = Vec.filled(4, 1, fx.Int32)
                a_frag = [[_cf4 for _ in range_constexpr(N_SUB)] for _ in range_constexpr(N_TILES_A)]
                bl_frag = [[_cf4 for _ in range_constexpr(N_SUB)] for _ in range_constexpr(N_TILES_BH)]
                br_frag = [[_cf4 for _ in range_constexpr(N_SUB)] for _ in range_constexpr(N_TILES_BH)]
            elif const_expr(_APREF):
                # A: use carried (read last iter -> latency hidden by last MFMA). Issue
                # NEXT iter's A read now (A_buf[(k+1)%3], landed) so it overlaps THIS MFMA.
                a_frag = a_carry
                if const_expr(k + 1 < KI):
                    a_carry = a_s2r.load(A_buf[(k + 1) % 3])
                bl_frag = b_s2r.load(BL_buf[cur_b])
                br_frag = b_s2r.load(BR_buf[cur_b])
            else:
                a_frag = a_s2r.load(A_buf[cur_a])
                bl_frag = b_s2r.load(BL_buf[cur_b])
                br_frag = b_s2r.load(BR_buf[cur_b]) if const_expr(HAS_BR) else None

            if const_expr(not HAS_BR):
                # BLOCK_N=128 single-slice (occ=2 path): A (NABUF-1)-ahead + B 1-ahead
                # FlyDSL G2S; single-slice MFMA (accL only). 32 accs=128 AGPR + ~96V
                # operands + 2-buf A -> V+A<=256 & LDS 80K/WG -> 2 waves/SIMD hides ds_read.
                if const_expr(has_b):
                    bl_g2s.load(BL_buf[nxt_b], BL_off + (k + 1) * KSTEP)
                if const_expr(has_a):
                    a_g2s.load(A_buf[wr_a], A_off + (k + AHEAD) * KSTEP)
                rocdl.s_setprio(1)
                if const_expr(_ASMM):
                    accL = mfma.call_packed_asm(a_frag, bl_frag, accL, sa, sbl, N_SUB)
                elif const_expr(PACKED):
                    accL = _mfma_packed(a_frag, bl_frag, accL, sa, sbl)
                else:
                    accL = mfma.call_subs(a_frag, bl_frag, accL, sa, sbl, N_SUB)
                rocdl.s_setprio(0)
                _VMCN = const_expr(int(__import__("os").environ.get("FP4_VMCN", str(N_LDS_STEPS_A))))
                if const_expr(_LEAN < 2):
                    wait_barrier(_VMCN if has_a else 0)
                continue

            if const_expr(_ASMM == 5):
                # HYBRID v3: cross-iter double-buffer. MFMA uses a_cur_pf (read last iter,
                # latency hidden by last MFMA). This asm reads NEXT iter's A (carried) +
                # THIS iter's B, both spread 1:1. G2S as in =3/=4.
                g2s = []
                if const_expr(has_b):
                    _bl = BL_off + fx.Int32((k + 1) * KSTEP)
                    _br = BR_off + fx.Int32((k + 1) * KSTEP)
                    for st in range_constexpr(N_LDS_STEPS_BH):
                        g2s.append((_m0(BL_buf[nxt_b], st), fx.Int32(gl_off_b[st]), rsrc_b, _bl))
                    for st in range_constexpr(N_LDS_STEPS_BH):
                        g2s.append((_m0(BR_buf[nxt_b], st), fx.Int32(gl_off_b[st]), rsrc_b, _br))
                if const_expr(has_a):
                    _ab = A_off + fx.Int32((k + AHEAD) * KSTEP)
                    for st in range_constexpr(N_LDS_STEPS_A):
                        g2s.append((_m0(A_buf[wr_a], st), fx.Int32(gl_off_a[st]), rsrc_a, _ab))
                # next iter's A buffer (for prefetch read); clamp at last iter.
                _na_buf = A_buf[(k + 1) % NABUF] if const_expr(k + 1 < KI) else A_buf[cur_a]
                an_base = [a_s2r.base_addr(_na_buf, s) for s in range_constexpr(N_SUB)]
                bl_base = [b_s2r.base_addr(BL_buf[cur_b], s) for s in range_constexpr(N_SUB)]
                br_base = [b_s2r.base_addr(BR_buf[cur_b], s) for s in range_constexpr(N_SUB)]
                rocdl.s_setprio(1)
                accL, accR, a_cur_pf = mfma.call_packed_asm2_g2s_dsr_pf(
                    a_cur_pf, an_base, bl_base, br_base, a_s2r.tile_stride, b_s2r.tile_stride,
                    accL, accR, sa, sbl, sbr, N_SUB, g2s)
                rocdl.s_setprio(0)
                _VMCN = const_expr(int(__import__("os").environ.get("FP4_VMCN", str(N_LDS_STEPS_A))))
                if const_expr(_LEAN < 2):
                    wait_barrier(_VMCN if has_a else 0)
                continue

            if const_expr(_ASMM == 4):
                # HYBRID v2: operand ds_read ALSO inside the asm (staggered lgkmcnt hides
                # the read latency behind the MFMA -> the 4915->5759 lever). G2S as in =3.
                g2s = []
                if const_expr(has_b):
                    _bl = BL_off + fx.Int32((k + 1) * KSTEP)
                    _br = BR_off + fx.Int32((k + 1) * KSTEP)
                    for st in range_constexpr(N_LDS_STEPS_BH):
                        g2s.append((_m0(BL_buf[nxt_b], st), fx.Int32(gl_off_b[st]), rsrc_b, _bl))
                    for st in range_constexpr(N_LDS_STEPS_BH):
                        g2s.append((_m0(BR_buf[nxt_b], st), fx.Int32(gl_off_b[st]), rsrc_b, _br))
                if const_expr(has_a):
                    _ab = A_off + fx.Int32((k + 2) * KSTEP)
                    for st in range_constexpr(N_LDS_STEPS_A):
                        g2s.append((_m0(A_buf[wr_a], st), fx.Int32(gl_off_a[st]), rsrc_a, _ab))
                a_base = [a_s2r.base_addr(A_buf[cur_a], s) for s in range_constexpr(N_SUB)]
                bl_base = [b_s2r.base_addr(BL_buf[cur_b], s) for s in range_constexpr(N_SUB)]
                br_base = [b_s2r.base_addr(BR_buf[cur_b], s) for s in range_constexpr(N_SUB)]
                rocdl.s_setprio(1)
                accL, accR = mfma.call_packed_asm2_g2s_dsr(
                    a_base, bl_base, br_base, a_s2r.tile_stride, b_s2r.tile_stride,
                    accL, accR, sa, sbl, sbr, N_SUB, g2s)
                rocdl.s_setprio(0)
                _VMCN = const_expr(int(__import__("os").environ.get("FP4_VMCN", str(N_LDS_STEPS_A))))
                if const_expr(_LEAN < 2):
                    wait_barrier(_VMCN if has_a else 0)
                continue

            if const_expr(_ASMM == 3):
                # HYBRID: G2S buffer_load_lds hand-interleaved INSIDE the MFMA asm cluster
                # (next-iter prefetch). No FlyDSL G2S ops -> I place every instruction.
                # ORDER: B[k+1] FIRST (oldest, 1-ahead -> end vmcnt drains it ready for next
                # iter), A[k+2] LAST (newest -> vmcnt(N_LDS_STEPS_A) keeps it in flight, it's
                # 2-ahead so drained next iter). FIFO keeps newest -> A stays, B drains.
                g2s = []
                if const_expr(has_b):
                    _bl = BL_off + fx.Int32((k + 1) * KSTEP)
                    _br = BR_off + fx.Int32((k + 1) * KSTEP)
                    for st in range_constexpr(N_LDS_STEPS_BH):
                        g2s.append((_m0(BL_buf[nxt_b], st), fx.Int32(gl_off_b[st]), rsrc_b, _bl))
                    for st in range_constexpr(N_LDS_STEPS_BH):
                        g2s.append((_m0(BR_buf[nxt_b], st), fx.Int32(gl_off_b[st]), rsrc_b, _br))
                if const_expr(has_a):
                    _ab = A_off + fx.Int32((k + 2) * KSTEP)
                    for st in range_constexpr(N_LDS_STEPS_A):
                        g2s.append((_m0(A_buf[wr_a], st), fx.Int32(gl_off_a[st]), rsrc_a, _ab))
                rocdl.s_setprio(1)
                accL, accR = mfma.call_packed_asm2_g2s(a_frag, bl_frag, br_frag, accL, accR,
                                                       sa, sbl, sbr, N_SUB, g2s)
                rocdl.s_setprio(0)
                _VMCN = const_expr(int(__import__("os").environ.get("FP4_VMCN", str(N_LDS_STEPS_A))))
                if const_expr(_LEAN < 2):
                    wait_barrier(_VMCN if has_a else 0)
                continue

            # PREFETCH before the 128-MFMA region so the MFMA hides the buffer_load latency.
            # B[k+1] before A[k+2] (A newest) -> end drain keeps A in flight, drains B.
            if const_expr(has_b):
                bl_g2s.load(BL_buf[nxt_b], BL_off + (k + 1) * KSTEP)
                br_g2s.load(BR_buf[nxt_b], BR_off + (k + 1) * KSTEP)
            if const_expr(has_a):
                a_g2s.load(A_buf[wr_a], A_off + (k + 2) * KSTEP)

            rocdl.s_setprio(1)
            if const_expr(_ASMM == 2):
                accL, accR = mfma.call_packed_asm2(a_frag, bl_frag, br_frag, accL, accR,
                                                   sa, sbl, sbr, N_SUB)
            elif const_expr(_ASMM):
                accL = mfma.call_packed_asm(a_frag, bl_frag, accL, sa, sbl, N_SUB)
                accR = mfma.call_packed_asm(a_frag, br_frag, accR, sa, sbr, N_SUB)
            elif const_expr(PACKED):
                accL = _mfma_packed(a_frag, bl_frag, accL, sa, sbl)
                accR = _mfma_packed(a_frag, br_frag, accR, sa, sbr)
            else:
                accL = mfma.call_subs(a_frag, bl_frag, accL, sa, sbl, N_SUB)
                accR = mfma.call_subs(a_frag, br_frag, accR, sa, sbr, N_SUB)
            rocdl.s_setprio(0)

            # END drain + barrier (loop's ONLY sync): B[k+1] landed during the MFMA above
            # (latency hidden -> cheap); keep A[k+2] (2-ahead slack). The s_barrier gives
            # next iter's reads visibility + WAR safety (overwrite of a buffer last read
            # 1-2 iters ago, globally done). FP4_VMCN tunes how many loads stay in flight.
            _VMCN = const_expr(int(__import__("os").environ.get("FP4_VMCN", str(N_LDS_STEPS_A))))
            _ENDM = const_expr(int(__import__("os").environ.get("FP4_ENDMODE", "0")))
            _vc = const_expr(_VMCN if has_a else 0)
            if const_expr(_LEAN < 2):
                if const_expr(_ENDM == 1):      # vmcnt-only (isolate: no workgroup barrier)
                    _llvm.inline_asm(res=None, operands_=[], asm_string=f"s_waitcnt vmcnt({_vc})",
                                     constraints="", has_side_effects=True)
                elif const_expr(_ENDM == 2):    # barrier-only (isolate: no vmcnt drain)
                    rocdl.s_barrier()
                elif const_expr(_ENDM == 3):    # vmcnt every iter (cheap) + barrier every 2 iters
                    _llvm.inline_asm(res=None, operands_=[], asm_string=f"s_waitcnt vmcnt({_vc})",
                                     constraints="", has_side_effects=True)
                    if const_expr((k % 2 == 1) or (not has_a)):
                        rocdl.s_barrier()
                else:
                    wait_barrier(_vc)

        base_row = block_m * BLOCK_M + wave_m_off
        base_col_l = block_n * BLOCK_N + wave_n_off
        base_col_r = block_n * BLOCK_N + LDS_BN_HALF + wave_n_off
        store_c.store(accL, base_row, base_col_l)
        if const_expr(HAS_BR):
            store_c.store(accR, base_row, base_col_r)

    # agpr-alloc=256 lets the backend place the MFMA accumulators in AGPR. NOTE: the
    # gluon `amdgpu-mfma-vgpr-form=false` knob is a no-op here — it is dropped before
    # LLVM IR (acc stays in the 256 ArchVGPR, AGPR=0, verified by profiling), so the
    # scaled-MFMA acc cannot be moved to AGPR via attributes in this build.
    # agpr-alloc sized to the live accs (256 for BN256's 64 accs, 128 for BN128's 32) so
    # the BN128 path leaves V+A<=256/wave for 2 waves/SIMD (occ=2).
    _AGPR_ALLOC = "256" if BLOCK_N == 256 else "128"
    _pt = {"passthrough": [["amdgpu-agpr-alloc", _AGPR_ALLOC]]} if agpr else {}

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
        # waves_per_eu=1 -> 1 wave/SIMD -> the full 512-VGPR file is one wave's, so the
        # 256-VGPR accumulator (accL+accR) + operands fit WITHOUT spill (default occ
        # heuristic caps VGPR at 256 => 92B scratch spill; profiling found this).
        kernel_gemm_4w(
            A, B_T, C, A_scale, B_scale, c_m, c_n,
            value_attrs={"rocdl.flat_work_group_size": "256,256",
                         "rocdl.waves_per_eu": OCC, **_pt},
        ).launch(grid=(grid_x, 1, 1), block=(256, 1, 1), stream=stream)

    # maxnreg (--amdgpu-num-vgpr): cap ArchVGPR so the 256-reg accumulator is forced
    # into AGPR (with agpr-alloc=256) instead of filling all 256 VGPR + spilling.
    if maxnreg:
        launch_gemm.compile_hints = {**getattr(launch_gemm, "compile_hints", {}), "maxnreg": maxnreg}
    # FP4_VGPRFORM=1: force AGPR-form MFMA (amdgpu-mfma-vgpr-form=false) to kill the
    # per-MFMA v_accvgpr_read/write shuffle (ISA showed 8164 of them = #1 issue-ceiling
    # overhead). Only meaningful via external LLVM (FLYDSL_COMPILE_LLVM_DIR) where the
    # cl-opt reaches codegen; bundled build drops it (see note above).
    import os as _os
    if _os.environ.get("FP4_VGPRFORM", "0") != "0":
        _lo = {"amdgpu-mfma-vgpr-form": False}
        _sched = _os.environ.get("FP4_SCHED", "")  # e.g. "max-ilp" / "max-memory-clause"
        if _sched:
            _lo["amdgpu-sched-strategy"] = _sched
        _bias = _os.environ.get("FP4_SBIAS", "")
        if _bias:
            _lo["amdgpu-schedule-metric-bias"] = int(_bias)
        launch_gemm.compile_hints = {**getattr(launch_gemm, "compile_hints", {}), "llvm_options": _lo}
    return launch_gemm
