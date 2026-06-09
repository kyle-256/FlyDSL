# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""MXFP4 dense GEMM (per-1x32 E8M0 block-scaled E2M1 fp4) for AMD CDNA4 (gfx950).

Correctness-first v0: A/B fp4 fragments loaded DIRECTLY gmem->reg per MFMA (no
LDS staging / no software pipeline) to validate the fp4 MFMA + scale path. Once
correct, an LDS-staged + pipelined version follows for perf.

fp4 layout: packed 2/byte (byte b = K[2b] low nibble | K[2b+1] high nibble).
A 16x16x128 mfma_scale (cbsz=4/blgp=4) contracts 128 fp4 = 4 micro-blocks of 32;
lane (g=lane//16, r=lane%16) provides row/col r, block g = 32 fp4 = 16 contiguous
bytes [k*64 + g*16 ..]. Operand is i32x8 with low 16B real + upper 16B zero.
Scale path is IDENTICAL to mxfp8 (per-1x32 E8M0, 4 blocks/mfma) -- reused.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as _llvm
from flydsl._mlir.dialects import scf as _scf
from flydsl.expr import arith, buffer_ops, const_expr, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec


def _raw(v):
    if not isinstance(v, ir.Value) and hasattr(v, "ir_value"):
        return v.ir_value()
    return v


def wait_lgkm():
    """Drain LDS reads (lgkmcnt 0) WITHOUT a WG barrier -- enforces S2R reads
    complete before a G2S buffer_load_lds overwrites the same LDS slot (WAR),
    while still letting the async G2S overlap the following MFMAs."""
    _llvm.inline_asm(res=None, operands_=[], asm_string="s_waitcnt lgkmcnt(0)",
                     constraints="", has_side_effects=True)


def wait_barrier_lgkm():
    """Drain LDS writes (lgkmcnt) then WG-sync — used after the padded 2-hop G2S
    (gmem->VGPR->ds_write) so all waves' ds_writes are visible before S2R reads.
    The stock buffer_load_lds completes on vmcnt; the 2-hop ds_write completes on
    lgkmcnt, so the vmcnt-based wait_barrier no longer guarantees LDS visibility."""
    _llvm.inline_asm(
        res=None,
        operands_=[],
        asm_string="s_waitcnt lgkmcnt(0)\ns_barrier",
        constraints="",
        has_side_effects=True,
    )


from kernels.fp8_gemm_utils import (
    G2SLoader,
    ceildiv,
    divmod,
    make_fp8_buffer_tensor,
    pack_i32x4_i32x8,
    wait_barrier,
)
from turbo.mxfp8_gemm_8wave import (
    ScaleBComb,
    ScaleS2R,
    StoreCPlain,
    preshuffle_scale,  # noqa: F401  (host helper, used by test)
    preshuffle_scale_b_comb,  # noqa: F401
)


class StoreCAtomic:
    """Split-K epilogue: atomic-add the FP32 accumulator into an FP32 scratch C.

    Each K-split's workgroup computes a partial sum for the same output tile and
    accumulates it into a zeroed FP32 buffer via native ``global_atomic_add_f32``
    (fast; scalar bf16 atomicrmw lowers to a slow + incorrect CAS loop on gfx950).
    The host casts the FP32 scratch -> BF16 after the launch. C here is the FP32
    scratch [M,N]. Raw ptr<1> has no HW OOB clamp, so each add is guarded by an
    scf.if on (col_valid & row_valid). Used only when split_k>1.
    """

    def __init__(self, C, c_rows, c_cols, c_idx_fn, n_tiles_a, n_tiles_b):
        self.c_rows = c_rows
        self.c_cols = c_cols
        self.lane_id = fx.thread_idx.x % 64
        self.c_idx_fn = c_idx_fn
        self.n_tiles_a = n_tiles_a
        self.n_tiles_b = n_tiles_b
        base_idx = buffer_ops.extract_base_index(C, address_space=1)
        self.base = _llvm.inttoptr(ir.Type.parse("!llvm.ptr<1>"), _raw(fx.Int64(base_idx)))

    def store(self, c_frag, base_row, base_col):
        for ti in range_constexpr(self.n_tiles_a):
            row = base_row + ti * 16 + (self.lane_id // 16) * 4
            for tj in range_constexpr(self.n_tiles_b):
                col = base_col + tj * 16 + self.lane_id % 16
                col_valid = col < fx.Int32(self.c_cols)
                vec_f32 = Vec(c_frag[self.c_idx_fn(ti, tj)])
                for i in range_constexpr(4):
                    r = row + i
                    valid = arith.andi(_raw(col_valid), _raw(r < fx.Int32(self.c_rows)))
                    if_op = _scf.IfOp(valid, [], has_else=False)
                    with ir.InsertionPoint(if_op.then_block):
                        c_index = r * fx.Int32(self.c_cols) + col
                        ptr = buffer_ops.get_element_ptr(self.base, byte_offset=_raw(c_index * fx.Int32(4)), elem_type=T.i8)
                        _llvm.AtomicRMWOp(
                            _llvm.AtomicBinOp.fadd, ptr, _raw(vec_f32[i]),
                            _llvm.AtomicOrdering.monotonic, syncscope="agent", alignment=4,
                        )
                        _scf.YieldOp([])


class MfmaScaleFp4:
    """16x16x128 f8f6f4 MFMA in fp4 mode (cbsz=4/blgp=4) with per-block E8M0 scales.

    asm=True emits the MFMA as an opaque inline-asm op (operands i32x4 a/b — fp4
    HW takes 16B, the i32x8 padding is only for the rocdl-intrinsic path). Opaque
    asm lets us hand-place MFMAs relative to ds_read/buffer_load (LLIR-scheduler
    style interleave) without the LLVM machine scheduler re-clustering them.
    """

    # operands: a,b,sa,sb,c ; $0=D(out,AGPR), $1=a,$2=b,$3=sa,$4=sb, C operand=$0
    # (in-place accumulate; c tied to output via "0"). The 4th asm slot is C, NOT
    # opsel — the intrinsic disasm shows literal "0" there only because first-iter
    # C==0; for accumulation it must be the accumulator register ($0).
    _ASM = "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, $0, $3, $4 op_sel_hi:[0,0,0] cbsz:4 blgp:4"
    _CONS = "=a,v,v,v,v,0"
    # first-iter zero-init: C==0 literal, NO tie -> fresh distinct AGPR per acc
    # (avoids shared-zero collision AND keeps all accumulators in AGPR, so the
    # k>0 "=a" tie never hits a VGPR accumulator -- the intrinsic-for-k0 path
    # mixed VGPR/AGPR which broke the tie).
    _ASM0 = "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, 0, $3, $4 op_sel_hi:[0,0,0] cbsz:4 blgp:4"
    _CONS0 = "=a,v,v,v,v"

    def __init__(self, n_tiles_a, n_tiles_b, asm=False, asm_se=False):
        self.res_ty = Vec.make_type(4, fx.Float32)
        self.zero_value = Vec.filled(4, 0.0, fx.Float32)
        self.n_tiles_a = n_tiles_a
        self.n_tiles_b = n_tiles_b
        self.asm = asm
        self.asm_se = asm_se  # has_side_effects on the asm MFMA (preserves program order for interleave)
        self._zero4 = Vec.filled(4, 0, fx.Int32)

    def idx(self, i, j):
        return i * self.n_tiles_b + j

    def _do(self, a, b, c, sa, sb, use_asm=None, first=False, pin=None):
        if use_asm is None:
            use_asm = self.asm
        if use_asm:
            from flydsl.expr.utils.arith import _to_raw

            # Pin the accumulator output to a specific AGPR range a[pin:pin+3] so
            # LLVM cannot coalesce/mix the inline-asm accumulators (the =a / VGPR
            # auto-alloc lets it merge them -> garbage on 8-wave 32-acc).
            out_c = "=a" if pin is None else "={a[%d:%d]}" % (pin, pin + 3)
            # a/b must be i32x4 (unpadded) for the asm operand width.
            if first:  # zero-init via C==0 literal, no accumulator tie
                return _llvm.inline_asm(
                    self.res_ty,
                    [_to_raw(a), _to_raw(b), _to_raw(sa), _to_raw(sb)],
                    self._ASM0,
                    out_c + ",v,v,v,v",
                    has_side_effects=self.asm_se,
                )
            return _llvm.inline_asm(
                self.res_ty,
                [_to_raw(a), _to_raw(b), _to_raw(sa), _to_raw(sb), _to_raw(c)],
                self._ASM,
                out_c + ",v,v,v,v,0",
                has_side_effects=self.asm_se,
            )
        # intrinsic path: needs i32x8 padded a/b. asm-mode S2R gives i32x4 -> pad.
        if self.asm:
            a = pack_i32x4_i32x8(Vec(a), self._zero4)
            b = pack_i32x4_i32x8(Vec(b), self._zero4)
        return rocdl.mfma_scale_f32_16x16x128_f8f6f4(self.res_ty, [a, b, c, 4, 4, 0, sa, 0, sb])

    def call(self, a, b, c, sa, sb, use_asm=None, first=False):
        for i in range_constexpr(self.n_tiles_a):
            for j in range_constexpr(self.n_tiles_b):
                c[self.idx(i, j)] = self._do(a[i], b[j], c[self.idx(i, j)], sa[i], sb[j], use_asm, first)
        return c

    def call_subs(self, a, b, c, sa, sb, n_sub, use_asm=None):
        """Accumulate over n_sub 128-K sub-blocks. a[i][s]/b[j][s] nested frags;
        sa[s]=n_tiles_a i32, sb[s]=n_tiles_b i32 (per-sub-block E8M0 scales)."""
        for s in range_constexpr(n_sub):
            a_s = [a[i][s] for i in range_constexpr(self.n_tiles_a)]
            b_s = [b[j][s] for j in range_constexpr(self.n_tiles_b)]
            c = self.call(a_s, b_s, c, sa[s], sb[s], use_asm)
        return c

    def call_one(self, a, b, c, i, j, sa, sb):
        """Single (i,j) scaled MFMA — for the 4-warp interleaved cluster (BLOCK_K=128,
        flat a[i]/b[j] frags, sa[i]/sb[j] per-tile E8M0 scale)."""
        return self._do(a[i], b[j], c[self.idx(i, j)], sa[i], sb[j])


class Fp4FragLoader:
    """Loads one wave's fp4 A (or B) fragments directly gmem->reg for a K-iter.

    Per (sub-tile i, lane (g,r)): 16 contiguous bytes (32 fp4 = block g of the
    128-fp4 K-chunk) at gmem byte offset row*(K/2) + k*64 + g*16, packed into
    i32x8 (low 16B real, upper 16B zero) for the fp4 MFMA operand.
    """

    def __init__(self, tensor, dim, K, n_tiles):
        self.Kbytes = K // 2  # packed fp4 bytes per row
        self.n_tiles = n_tiles
        self.lane16 = fx.thread_idx.x % 16
        self.g = (fx.thread_idx.x % 64) // 16
        nbytes = dim * self.Kbytes
        self.rsrc = buffer_ops.create_buffer_resource(tensor, max_size=False, num_records_bytes=nbytes)
        self.zero4 = Vec.filled(4, 0, fx.Int32)

    def load(self, base_row, k):
        """base_row: global row/col base (multiple of 16) for this (region, wave)."""
        out = []
        for i in range_constexpr(self.n_tiles):
            row = base_row + (i * 16) + self.lane16
            # byte offset = row*Kbytes + k*64 + g*16 ; i32-elem offset = /4
            byte_off = row * self.Kbytes + k * 64 + self.g * 16
            v4 = Vec(buffer_ops.buffer_load(self.rsrc, byte_off // 4, vec_width=4, dtype=T.i32))
            out.append(pack_i32x4_i32x8(v4, self.zero4))
        return out


def fp4_g2s_offsets(lane_id, wave_id, K, n_steps, bytes_per_row):
    """Per-lane gmem byte offsets for fp4 G2S (identity LDS layout, no swizzle).

    A K-iter row is BLOCK_K fp4 == bytes_per_row packed bytes == (bpr/16) lanes
    x 16-byte copies. lanes_per_row lanes cover one row, 64 lanes cover
    64/lanes_per_row rows, n_waves waves cover that x n_waves rows per step. Lane
    L (wave w) copies 16 bytes from gmem [row*(K/2)+chunk*16] into its contiguous
    LDS slot; the contiguous slot (w*1024 + L*16) algebraically equals
    row*bpr + chunk*16, so S2R reads it back at identity row*bpr + g*16
    (proof: (L//lpr)*bpr + (L%lpr)*16 == L*16 since bpr == lpr*16).
    """
    n_waves = fx.block_dim.x // 64
    lpr = bytes_per_row // 16  # lanes per row
    rows_per_step = 64 // lpr
    offs = []
    for r in range_constexpr(n_steps):
        row = lane_id // lpr + wave_id * rows_per_step + r * (n_waves * rows_per_step)
        chunk = lane_id % lpr
        offs.append(row * (K // 2) + chunk * 16)
    return offs


def recommend_config(M, N, K):
    """Data-driven production config for dense mxfp4 (mode=pipe) on MI355X.

    Returns (BLOCK_M, BLOCK_N, group_m, group_n). Three verified-solid levers:

    - BLOCK_N = 128 for grid-underfilled (narrow-N kv) shapes: when the BLOCK_N=256
      grid has fewer tiles than the 256 active CUs, halving N doubles the N-tiles and
      fills the grid -> occupancy win (kv M=4096 +5.9%, M=8192 +0.1%). The fp4 pipe
      BN128 path is det0-clean (300-run maxdiff=0, per-region B-scale w/ wait(1)
      combined-G2S). This is NOT the discredited BLOCK_M=128 path (that one raced) --
      N-tiling is a different lever and is correctness-verified. group_n stays 0 here
      (kv nb at BN128 = 8 << 96).
    - group_n = nb//8 (bands ~= num_xcds) for nb>=96 (N>=24576): reliable big-N L2
      win, e.g. Llama 70B gate/up N=28672 = +8% (SNR bit-exact, det OK). Narrower N
      (11008/8192) showed only run-to-run noise -> left at 0.

    BLOCK_M stays 256 always: BLOCK_M=128 doubled the grid on narrow-N but is NOT
    correct on this fp4 pipe (SNR -3dB + nondeterministic) -> latent race; do not use.

    - group_m = 2 for the narrow-N grid-underfill (kv) regime (block_n==128). The
      per-shape autotune (round-26) found the static group_m=4 GROUP_M tiling is a
      per-shape config error for kv: with only ~8 N-tiles, the GROUP_M=4 M-clustering
      hurts XCD/L2 locality. group_m=2 is a robust +6.9% on kv M=8192 (3-trial:
      gm=4 [2549,2578,2624] vs gm=2 [2748,2767,2770], full separation) and neutral
      on kv M=4096 (+0.4%, no regress). Pure tile->CU permutation => bit-exact + det0.
      Wide-N shapes (block_n==256) keep group_m=4 (autotune confirmed gm sweep there
      is run-to-run noise; the band group_n lever owns their L2 locality).
    """
    NUM_CUS = 256
    tiles_256 = ((M + 255) // 256) * ((N + 255) // 256)
    block_n = 128 if tiles_256 < NUM_CUS else 256
    nb = N // block_n
    # round-61: big-K bulk ("down" projections, K>=11008) ALSO benefit from the 2D
    # band swizzle even at modest nb (16-32), not just very-wide-N (nb>=96). Joint
    # (group_m x group_n) reliable-interleaved sweep + high-T fresh-data confirm:
    # 7B down (N=4096,K=11008) gn=2 = +7.9%/+4.1% (M4096/M8192, 15/15); 70B down
    # (N=8192,K=28672) gn=4 = +1.9%/+0.9% (12-15/15). Overturns r28's "big-K band
    # hurts" (that was a 1-D gn sweep at fixed gm / non-interleaved = unreliable);
    # the skill's "2D band general, big-K also +1%" was right. Big-K = larger
    # per-tile B-stripe L2-reuse deficit. K>=11008 selects exactly the two down
    # shapes; square/q-o (K<=8192) stay gn0 (joint sweep confirmed noise there).
    # round-76 (aligned nx x gn joint sweep, GOLD T=15 fresh-data interleaved): the
    # "down" (K>>N) regime wants an ABSOLUTE band width ~4, not nb//8. 7B down
    # (nb=16) nb//8=2 -> gn=4 confirmed +1.0%/+1.5% (M4096/M8192, 15/15 both); 70B
    # down (nb=32) already nb//8=4 (byte-identical). big-N (nb>=96) keeps nb//8 (=14
    # for 70B gate/up, #bands=#XCD). r61's nb//8 down rule under-set 7B down's band.
    if nb >= 96:
        group_n = nb // 8
    elif K >= 11008:
        group_n = 4
    else:
        group_n = 0
    group_m = 2 if block_n == 128 else 4
    # B6 (r6_4): BLOCK_M=192 for the grid-underfilled kv regime where the BM256/BN128
    # grid still leaves CUs idle. ceil(M/256)*ceil(N/128) WG at BM256: kv M4096=128wg
    # (<256, half the CUs idle); BM192 -> 22 M-tiles vs 16 = 176wg at 0.75x per-WG MFMA
    # -> +6.0% (r6_3 interleaved 10/10, det0, pipe narrow-A-G2S SNR 336dB). kv M8192 is
    # already 256wg-full at BM256 -> BM192 (344wg) overshoots & regresses -18%, so gate
    # on the underfill. NOT the discredited BLOCK_M=128 (that raced); BM192 is the
    # quadrant's smallest correct M-tile (N_TILES_A=3, narrow-A-G2S clamp-wave).
    # B10 (r10, round-54): among {128,192,256}, pick the SMALLEST BLOCK_M whose grid
    # does not overshoot NUM_CUS (more M-tiles = fuller grid) for the underfilled kv
    # regime (block_n==128). kv M4096 -> BM128 (256wg FULL, +1.9% over BM192/176wg,
    # interleaved 10/10 thermal-matched); kv M8192 -> BM256 (256wg-full; BM128=512wg
    # / BM192=344wg both overshoot -> regress). BM128 is now CORRECT + DET0 on the
    # production pipe (staged+pipe SNR 55.6, pipe det0 2pass x 200run) — the
    # pre-campaign "BM128 races (SNR -3dB nondet)" death-list entry PRE-DATED the
    # narrow-A-G2S clamp-wave (r6_3, BM128 reuses it: LDS_BLOCK_M=64<128 -> A_NARROW,
    # NW_A_ACTIVE=4) and wait_barrier(1) (r_k7) race fixes; both falsify that entry.
    # Wide-N (block_n==256) always BM256. Supersedes the r6_4 BM192 kv route.
    block_m = 256
    if block_n == 128:
        for _bm in (128, 192):
            if ((M + _bm - 1) // _bm) * ((N + block_n - 1) // block_n) <= NUM_CUS:
                block_m = _bm
                break
    # B10 (r55): the BM128 kv grid (32 M-tiles x 8 N-tiles = 256 tiles) prefers a
    # WIDER GROUP_M super-block than the BM192 grid did. r26's gm2 was tuned for the
    # BM192 22-M-tile grid; re-sweeping for BM128 (interleaved 10/10, kv M4096):
    # gm8 +1.3% vs gm2, gm1 +0.5%, gm4 -0.2% -> gm8 robust optimum. Pure tile->CU
    # permutation (bit-exact, det-neutral). kv M8192 stays BM256/gm2 (r26, unchanged).
    if block_m == 128:
        group_m = 8
    return block_m, block_n, group_m, group_n


def grouped_xcd_pid(pid, c_m, c_n, BLOCK_M, BLOCK_N, group_m=4, num_xcds=8, group_n=0):
    """Map block_idx -> (block_m, block_n) with XCD-aware remap + GROUP_M tiling
    for L2 locality (mirrors the gfx950 a4w4 reference). Pure index math, no data
    path change. Exact bijection when total_tiles % num_xcds == 0 (holds for the
    bench shapes); falls back safely otherwise.

    group_n>0 enables a 2D super-block (band) swizzle on top: N is split into
    vertical bands of width ``group_n`` N-tiles, and the classic GROUP_M tiling
    runs *within* each band. This locks an A-slab (group_m tiles) AND a B-slab
    (group_n tiles) into L2 simultaneously -> A reused group_n*, B reused group_m*.
    The proven big-N L2 lever (1D GROUP_M alone does not help wide-N shapes;
    see flydsl-fp8-gemm-tuning skill: big-N L2 51%->57.5%, +12%). group_n=0
    keeps the exact prior 1D XCD+GROUP_M behaviour.
    """
    from flydsl.expr import arith

    num_pid_m = ceildiv(c_m, BLOCK_M)
    num_pid_n = ceildiv(c_n, BLOCK_N)
    total = num_pid_m * num_pid_n
    pids_per_xcd = (total + num_xcds - 1) // num_xcds
    pid_r = (pid % num_xcds) * pids_per_xcd + pid // num_xcds
    pid_r = arith.select(pid_r < total, pid_r, pid)

    if group_n and group_n > 0:
        # 2D band: split N into full bands of width group_n + one remainder band.
        band_tiles = num_pid_m * group_n            # tiles in one full band
        n_full_bands = num_pid_n // group_n
        full_region = n_full_bands * band_tiles     # pids covered by full bands
        in_full = pid_r < full_region
        # full-band branch
        band_id = pid_r // band_tiles
        local_f = pid_r % band_tiles
        nbase_f = band_id * group_n
        bw_f = fx.Int32(group_n)
        # remainder-band branch (width = num_pid_n - n_full_bands*group_n)
        rem = num_pid_n - n_full_bands * group_n
        local_r = pid_r - full_region
        nbase_r = n_full_bands * group_n
        bw_r = arith.select(rem < fx.Int32(1), fx.Int32(1), rem)  # avoid /0 in dead branch
        local = arith.select(in_full, local_f, local_r)
        nbase = arith.select(in_full, nbase_f, nbase_r)
        bw = arith.select(in_full, bw_f, bw_r)
        # classic GROUP_M tiling inside (num_pid_m x bw)
        num_in_group = group_m * bw
        group_id = local // num_in_group
        first_m = group_id * group_m
        gsz = num_pid_m - first_m
        gsz = arith.select(gsz < fx.Int32(group_m), gsz, fx.Int32(group_m))
        inner = local % num_in_group
        block_m = first_m + inner % gsz
        block_n = nbase + inner // gsz
        return block_m, block_n

    num_in_group = group_m * num_pid_n
    group_id = pid_r // num_in_group
    first_m = group_id * group_m
    gsz = num_pid_m - first_m
    gsz = arith.select(gsz < fx.Int32(group_m), gsz, fx.Int32(group_m))
    inner = pid_r % num_in_group
    block_m = first_m + inner % gsz
    block_n = inner // gsz
    return block_m, block_n


class PaddedG2SLoader:
    """Custom 2-hop fp4 G2S (gmem -> VGPR -> padded LDS). Writes each lane's 16B
    to LDS element index row*lds_row_stride + chunk*16. A padded row stride
    (> bytes_per_row) shifts consecutive rows across LDS banks (mirrors the
    competitor's PaddedSharedLayout), killing the bank conflict that the stock
    contiguous buffer_load_lds atom forces (it can only write identity stride).
    Layout stays identity-with-padding so S2R reads at the same row*stride+col*16.
    """

    def __init__(self, tensor, dim, K, lane_id, wave_id, n_steps, bytes_per_row, lds_row_stride):
        self.K2 = K // 2
        self.n_steps = n_steps
        nbytes = dim * self.K2
        self.rsrc = buffer_ops.create_buffer_resource(tensor, max_size=False, num_records_bytes=nbytes)
        n_waves = fx.block_dim.x // 64
        lpr = bytes_per_row // 16
        rows_per_step = 64 // lpr
        self.gmem_base = []
        self.lds_idx = []
        for r in range_constexpr(n_steps):
            row = lane_id // lpr + wave_id * rows_per_step + r * (n_waves * rows_per_step)
            chunk = lane_id % lpr
            self.gmem_base.append(row * self.K2 + chunk * 16)
            self.lds_idx.append(row * lds_row_stride + chunk * 16)

    def load(self, lds_buf, k_offset):
        for r in range_constexpr(self.n_steps):
            byte_off = self.gmem_base[r] + k_offset
            v = Vec(buffer_ops.buffer_load(self.rsrc, byte_off // 4, vec_width=4, dtype=T.i32))
            ptr_off = fx.add_offset(lds_buf.ptr, fx.make_int_tuple(self.lds_idx[r]))
            i8_iter = fx.recast_iter(fx.Uint8, ptr_off)
            view = fx.make_view(i8_iter, fx.make_layout(16, 1))
            view.store(v.bitcast(fx.Uint8))


class S2RLoaderFp4:
    """LDS->reg fp4 fragment loader (identity LDS, bytes_per_row K-iter rows).

    A K-iter spans n_sub == BLOCK_K/128 128-K sub-blocks; each sub-block s is one
    16x16x128 MFMA. Lane (g=lane//16, r=lane%16) reads, per sub-block, its block
    g == 16 contiguous packed bytes (32 fp4) at row*bpr + s*64 + g*16, bitcast to
    i32x4 then padded to i32x8 (low 16B real, upper zero) for the cbsz=4 operand.
    ``load`` returns frag[tile][sub] (nested).
    """

    def __init__(self, wave_idx, n_tiles, n_sub, bytes_per_row, row_stride=None, pad=True):
        self.lane16 = fx.thread_idx.x % 16
        self.g = (fx.thread_idx.x % 64) // 16
        self.wave_idx = wave_idx
        self.n_tiles = n_tiles
        self.n_sub = n_sub
        self.bpr = bytes_per_row
        self.row_stride = row_stride if row_stride is not None else bytes_per_row
        self.pad = pad  # False -> return raw i32x4 (for inline-asm MFMA; fp4 HW needs only 16B)
        self.zero4 = Vec.filled(4, 0, fx.Int32)

    def _load16(self, lds_src, off_bytes):
        off_tup = fx.make_int_tuple(off_bytes)
        ptr_off = fx.add_offset(lds_src.ptr, off_tup)
        i8_iter = fx.recast_iter(fx.Uint8, ptr_off)
        view = fx.make_view(i8_iter, fx.make_layout(16, 1))
        return view.load()

    def load(self, lds_src):
        frag = []
        for i in range_constexpr(self.n_tiles):
            row = self.wave_idx * (self.n_tiles * 16) + i * 16 + self.lane16
            subs = []
            for s in range_constexpr(self.n_sub):
                off = row * self.row_stride + s * 64 + self.g * 16
                v4 = self._load16(lds_src, off).bitcast(fx.Int32)
                subs.append(v4 if not self.pad else pack_i32x4_i32x8(v4, self.zero4))
            frag.append(subs)
        return frag


def il_mma(mfma, quads, prefetch, ua, n_ta, n_tb, per=8, first=False, pinned=False, sched_mm=0, sched_dsrd_n=0):
    """One K-iter's MFMAs (len(quads) quadrants x n_ta*n_tb tiles) with the
    prefetch G2S load_one's spread 1-per-``per``-MFMAs. Module-level (no closures)
    so FlyDSL's kernel AST transform doesn't choke. Mutates each quad's c list.
    quads = [(c, a, b, sca, scb), ...]; sca[i]/scb[j] flat per-tile E8M0 scales.
    first=True -> zero-init accumulators (C==0 asm, no tie).
    pinned=True -> pin accumulator q's output to AGPR a[(qg)*4:(qg)*4+3] (qg =
    global accumulator index) so LLVM can't coalesce them."""
    li = 0
    cnt = 0
    qg = 0
    n_pf = len(prefetch)
    for cc, aa, bb, sca, scb in quads:
        for i in range_constexpr(n_ta):
            for j in range_constexpr(n_tb):
                pin = (qg * 4) if pinned else None
                cc[mfma.idx(i, j)] = mfma._do(aa[i][0], bb[j][0], cc[mfma.idx(i, j)], sca[i], scb[j], ua, first, pin)
                qg += 1
                cnt += 1
                if n_pf and cnt % per == 0 and li < n_pf:
                    g, ld, ko, st = prefetch[li]
                    g.load_one(ld, ko, st)
                    li += 1
    while li < n_pf:
        g, ld, ko, st = prefetch[li]
        g.load_one(ld, ko, st)
        li += 1
    if sched_mm:
        # Manual LLIR-scheduler cadence: tell the LLVM scheduler to interleave the
        # region's ds_read (S2R) + vmem (G2S/scale) among the MFMAs at the fp4
        # throughput rate (~sched_mm MFMA per memory op), so operands land just in
        # time and the matrix unit never starves -- what the competitor's custom
        # LLIR pass does, expressed via the native sched_group_barrier intrinsic.
        n_mfma = len(quads) * n_ta * n_tb
        n_mem = sched_dsrd_n + n_pf
        if n_mem > 0:
            base = n_mfma // n_mem
            rem = n_mfma - base * n_mem
            dsrd_left = sched_dsrd_n
            vmem_left = n_pf
            for gi in range_constexpr(n_mem):
                grp = base + (1 if gi < rem else 0)
                if grp > 0:
                    rocdl.sched_mfma(grp)
                if dsrd_left > 0:
                    rocdl.sched_dsrd(1)
                    dsrd_left -= 1
                else:
                    rocdl.sched_vmem(1)
                    vmem_left -= 1


class ScaleBRegion:
    """Per-N-region B E8M0 scale loader for BLOCK_N=128 (N_TILES_B sub-tiles per
    64-wide N-half). Mirrors mxfp8 ScaleS2R layout (host = preshuffle_scale(b_sc,
    K, n_tiles)) but is robust for n_tiles==1, where buffer_load(vec_width=1)
    returns a scalar (ScaleS2R wraps it in Vec(...) which crashes for scalars).
    Returns a list of n_tiles raw i32 scale operands (one per B sub-tile)."""

    def __init__(self, sp_tensor, dim, K, n_tiles):
        self.K128 = K // 128
        self.n_tiles = n_tiles
        self.group_span = 16 * n_tiles
        self.lane = fx.thread_idx.x % 64
        nbytes = (dim // self.group_span) * self.K128 * 64 * n_tiles * 4
        self.rsrc = buffer_ops.create_buffer_resource(sp_tensor, max_size=False, num_records_bytes=nbytes)

    def load(self, base, k):
        grp = base // self.group_span
        idx = ((grp * self.K128 + k) * 64 + self.lane) * self.n_tiles
        if self.n_tiles == 1:
            v = buffer_ops.buffer_load(self.rsrc, idx, vec_width=1, dtype=T.i32)
            return [_raw(v)]
        v = Vec(buffer_ops.buffer_load(self.rsrc, idx, vec_width=self.n_tiles, dtype=T.i32))
        return [v[i].ir_value() for i in range_constexpr(self.n_tiles)]

    def load_halves(self, base, lds_block_n, k):
        """Uniform N-half interface: the two BLOCK_N=128 N-halves (b0 at base,
        b1 at base+lds_block_n) each carry N_TILES_B scales. Returns (b0, b1).
        Branch-free vs ScaleBComb (same method name) so pipe `_sb` needs no
        kernel-body `if` (avoids the FlyDSL AST per-branch-fn trace quirk)."""
        return self.load(base, k), self.load(base + lds_block_n, k)


def compile_mxfp4_gemm_8w(
    *,
    K: int,
    BLOCK_M: int = 256,
    BLOCK_N: int = 256,
    mode: str = "direct",
    block_k: int = 128,
    padded: bool = False,
    pad_bytes: int = 16,
    asm_mfma: bool = False,
    asm_se: bool = False,
    frag_pad: bool = True,
    sched: bool = False,
    iglp: bool = False,
    group_m: int = 4,
    num_xcds: int = 8,
    group_n: int = 0,
    split_k: int = 1,
    wave_topo: str = "2x4",
):
    # block_k: logical fp4 contracted per K-iter. The 16x16x128 MFMA always does
    # 128 K, so a K-iter spans N_SUB == block_k/128 sub-block MFMAs per accumulator.
    # block_k only affects staged/pipe (direct hardwires 128).
    BLOCK_K = block_k if mode in ("staged", "pipe", "pipeh") else 128
    # B6: BLOCK_M=192 (small-M occupancy for grid-underfilled kv) needs only
    # BLOCK_M % 64 == 0 (N_TILES_A = BLOCK_M//64 integral) + (BLOCK_M//2) % 16 == 0
    # (LDS region a 16-row-wave granular). BM192 -> N_TILES_A=3, LDS_BLOCK_M=96.
    assert BLOCK_M >= 128 and BLOCK_M % 64 == 0 and (BLOCK_M // 2) % 16 == 0
    # B4 (wave_topo="4x2", 4 M-waves x 2 N-waves) makes BLOCK_N=64 expressible
    # (N_TILES_B = BLOCK_N//64 = 1); the default 2x4 topology floors BLOCK_N at 128.
    assert wave_topo in ("2x4", "4x2")
    if wave_topo == "4x2":
        assert BLOCK_N >= 64 and BLOCK_N % 64 == 0
    else:
        assert BLOCK_N >= 128 and BLOCK_N % 128 == 0
    assert BLOCK_K % 128 == 0 and K % BLOCK_K == 0

    K_ITERS = K // BLOCK_K
    assert split_k >= 1 and K_ITERS % split_k == 0, f"split_k {split_k} must divide K_ITERS {K_ITERS}"
    KI = K_ITERS // split_k  # per-split K-iterations (== K_ITERS when split_k==1)
    N_SUB = BLOCK_K // 128  # 128-K MFMA sub-blocks per K-iter
    BPR = BLOCK_K // 2  # packed-fp4 bytes per K-iter row in LDS
    KSTEP = BPR  # gmem byte stride per K-iter
    # Wave topology: 2x4 = 2 M-waves x 4 N-waves (default); 4x2 = 4 M-waves x 2
    # N-waves. NW_N = #N-waves (the kernel-body divisor: wave_m=wave_id//NW_N,
    # wave_n=wave_id%NW_N). Region halving (c00/c01/c10/c11, LDS_BLOCK_*) is
    # topology-independent; only the per-wave tile counts flip.
    if wave_topo == "4x2":
        NW_N = 2
        N_TILES_A = BLOCK_M // 128
        N_TILES_B = BLOCK_N // 64
    else:
        NW_N = 4
        N_TILES_A = BLOCK_M // 64
        N_TILES_B = BLOCK_N // 128
    N_ACCUMS = N_TILES_A * N_TILES_B
    LDS_BLOCK_M = BLOCK_M // 2
    LDS_BLOCK_N = BLOCK_N // 2
    SA_TILES = N_TILES_A
    SB_TILES = N_TILES_B
    # Combined-B scale (ScaleBComb: one dwordx4 = 4 scales = 2 N-halves x N_TILES_B==2)
    # is only valid for BLOCK_N==256. BLOCK_N==128 -> N_TILES_B==1: each of the two
    # 64-wide N-halves (b0/b1) needs 1 scale -> use the generic per-region ScaleS2R
    # (n_tiles=N_TILES_B) with host preshuffle_scale(b_sc, K, N_TILES_B). r_k1 wires
    # this in the `direct` mode only; pipe/staged/il keep the BLOCK_N==256 path.
    B_COMB = BLOCK_N >= 256

    # G2S row coverage per step == n_waves * (64 / (BPR/16)) rows.
    _ROWS_PER_STEP = 64 // (BPR // 16) * (512 // 64)  # lanes_per_row -> rows/step * n_waves
    N_LDS_STEPS_A = LDS_BLOCK_M // _ROWS_PER_STEP
    N_LDS_STEPS_B = LDS_BLOCK_N // _ROWS_PER_STEP
    # Combined-128 B G2S for BLOCK_N<256 (BN128): each 64-wide N half-region is
    # < _ROWS_PER_STEP(=128) so N_LDS_STEPS_B==0 (G2S empty). The two halves
    # (b_lds0,b_lds1) are LDS-adjacent, so ONE G2S over the full BLOCK_N rows
    # fills both (rows 0..63 -> b_lds0, 64..127 -> b_lds1, identity layout).
    N_LDS_STEPS_B_FULL = BLOCK_N // _ROWS_PER_STEP
    # BN < _ROWS_PER_STEP (e.g. BN64=64 < 128): the whole B tile is narrower than
    # one cooperative G2S step, so N_LDS_STEPS_B_FULL==0 (B never loads). Fix:
    # a partial-wave combined G2S -- only the first NW_B_ACTIVE waves (16 rows
    # each) issue one step, filling b_lds0 + the LDS-adjacent b_lds1. Waves
    # >= NW_B_ACTIVE are predicated out (no over-read past the B region, no LDS OOB).
    B_NARROW = BLOCK_N < _ROWS_PER_STEP
    NW_B_ACTIVE = BLOCK_N // 16  # 16 B rows per wave (lanes_per_row=4 -> 64/4)
    # B6: A analog of the narrow-B clamp-wave G2S. BM192 -> LDS_BLOCK_M=96 < step
    # (128) -> N_LDS_STEPS_A==0 (A G2S empty). Fix: clamp-wave combined G2S per
    # 96-row A region (NW_A_ACTIVE=6 waves x 16 rows = 96; waves >= 6 redundantly
    # reload the last 16-row region -> in-bounds, idempotent, det-safe). Mirror of
    # the round-19/20 narrow-B fix, applied per-region (a_lds0, a_lds1) instead of
    # the combined-adjacent B trick.
    A_NARROW = LDS_BLOCK_M < _ROWS_PER_STEP
    NW_A_ACTIVE = LDS_BLOCK_M // 16  # 16 A rows per wave
    # Padded LDS: row stride > BPR shifts rows across banks (kills bank conflict).
    # pad_bytes must be a multiple of 16 to keep ds_read_b128 16B-aligned.
    LDS_ROW_STRIDE = BPR + (pad_bytes if padded else 0)
    a_lds_size = LDS_BLOCK_M * LDS_ROW_STRIDE  # packed bytes per region buffer
    b_lds_size = LDS_BLOCK_N * LDS_ROW_STRIDE
    K2 = K // 2  # packed-fp4 row stride in bytes

    @fx.struct
    class SharedStorageFp4:
        A_lds0: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        A_lds1: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        B_lds0: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
        B_lds1: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]

    @flyc.kernel(known_block_size=[512, 1, 1])
    def kernel_gemm_staged(
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

        lds = fx.SharedAllocator().allocate(SharedStorageFp4).peek()
        a_lds0 = lds.A_lds0
        a_lds1 = lds.A_lds1
        b_lds0 = lds.B_lds0
        b_lds1 = lds.B_lds1

        lane_id = fx.thread_idx.x % 64
        wave_id = fx.thread_idx.x // 64
        wave_m = wave_id // NW_N
        wave_n = wave_id % NW_N
        block_m, block_n = divmod(fx.block_idx.x, n_blocks)

        gA = make_fp8_buffer_tensor(A, F8_IR_t)
        gB = make_fp8_buffer_tensor(B_T, F8_IR_t)
        a_div = fx.logical_divide(gA, fx.make_layout(1, 1))
        b_div = fx.logical_divide(gB, fx.make_layout(1, 1))

        mfma = MfmaScaleFp4(N_TILES_A, N_TILES_B, asm=asm_mfma, asm_se=asm_se)
        if const_expr(padded):
            a_g2s = PaddedG2SLoader(A, c_m, K, lane_id, wave_id, N_LDS_STEPS_A, BPR, LDS_ROW_STRIDE)
            b_g2s = PaddedG2SLoader(B_T, c_n, K, lane_id, wave_id, N_LDS_STEPS_B, BPR, LDS_ROW_STRIDE)
        else:
            gl_off_a = fp4_g2s_offsets(lane_id, wave_id, K, N_LDS_STEPS_A, BPR)
            gl_off_b = fp4_g2s_offsets(lane_id, wave_id, K, N_LDS_STEPS_B, BPR)
            gl_off_b_full = fp4_g2s_offsets(lane_id, wave_id, K, N_LDS_STEPS_B_FULL, BPR)
            a_g2s = G2SLoader(a_div, gl_off_a, N_LDS_STEPS_A, F8_IR_t, wave_id)
            b_g2s = G2SLoader(b_div, gl_off_b, N_LDS_STEPS_B, F8_IR_t, wave_id)
            b_g2s_full = G2SLoader(b_div, gl_off_b_full, N_LDS_STEPS_B_FULL, F8_IR_t, wave_id)
            # B4 narrow-B G2S (BLOCK_N < step): one combined G2S step where the
            # wave index is clamped to [0, NW_B_ACTIVE-1]. Waves >= NW_B_ACTIVE
            # redundantly reload the last valid 16-row region (same gmem src + same
            # LDS dst -> idempotent, det-safe) instead of writing past the B region.
            # All 8 waves issue unconditionally -> no dynamic per-wave `if` (FlyDSL
            # traces stateful objects referenced in a dynamic-if branch as MLIR
            # state and raises TypeError).
            if const_expr(B_NARROW):
                _wid_b = fx.Int32(wave_id)
                eff_wave_b = (_wid_b < fx.Int32(NW_B_ACTIVE)).select(_wid_b, fx.Int32(max(NW_B_ACTIVE - 1, 0)))
                gl_off_b_narrow = fp4_g2s_offsets(lane_id, eff_wave_b, K, 1, BPR)
                b_g2s_narrow = G2SLoader(b_div, gl_off_b_narrow, 1, F8_IR_t, eff_wave_b)
            if const_expr(A_NARROW):
                _wid_a = fx.Int32(wave_id)
                eff_wave_a = (_wid_a < fx.Int32(NW_A_ACTIVE)).select(_wid_a, fx.Int32(max(NW_A_ACTIVE - 1, 0)))
                gl_off_a_narrow = fp4_g2s_offsets(lane_id, eff_wave_a, K, 1, BPR)
                a_g2s_narrow = G2SLoader(a_div, gl_off_a_narrow, 1, F8_IR_t, eff_wave_a)
        a_s2r = S2RLoaderFp4(wave_m, N_TILES_A, N_SUB, BPR, LDS_ROW_STRIDE, pad=(not asm_mfma) and frag_pad)
        b_s2r = S2RLoaderFp4(wave_n, N_TILES_B, N_SUB, BPR, LDS_ROW_STRIDE, pad=(not asm_mfma) and frag_pad)
        # A-scale num_records must cover the rows the kernel addresses =
        # ceil(c_m/BLOCK_M)*BLOCK_M (the padded extent). ScaleS2R floors
        # dim//group_span, so pass a group_span-ceil'd dim. For aligned M
        # (16*SA_TILES | c_m, true for all BM256 production shapes) this == c_m
        # (byte-identical); only BM192 (16*3=48 ∤ 4096) pads, covering the edge
        # M-tile's scale group (else valid rows 4080-4095 clamp to scale 0 -> SNR 24).
        _sa_q = 16 * SA_TILES
        sa_s2r = ScaleS2R(A_scale, ((c_m + _sa_q - 1) // _sa_q) * _sa_q, K, SA_TILES)
        sb_s2r = ScaleBComb(B_scale, c_n, K) if B_COMB else ScaleBRegion(B_scale, c_n, K, N_TILES_B)
        store_c = (StoreCAtomic if split_k > 1 else StoreCPlain)(C, c_m, c_n, mfma.idx, N_TILES_A, N_TILES_B)

        wave_m_offset = wave_m * (N_TILES_A * 16)
        wave_n_offset = wave_n * (N_TILES_B * 16)
        sa_base0 = fx.Int32(block_m * BLOCK_M + wave_m_offset)
        sa_base1 = sa_base0 + fx.Int32(LDS_BLOCK_M)
        sb_base0 = fx.Int32(block_n * BLOCK_N + wave_n_offset)

        # Per-region gmem byte bases (packed fp4): row * K2.
        A0_off = block_m * BLOCK_M * K2
        A1_off = (block_m * BLOCK_M + LDS_BLOCK_M) * K2
        B0_off = block_n * BLOCK_N * K2
        B1_off = (block_n * BLOCK_N + LDS_BLOCK_N) * K2

        c00 = [mfma.zero_value] * N_ACCUMS
        c01 = [mfma.zero_value] * N_ACCUMS
        c10 = [mfma.zero_value] * N_ACCUMS
        c11 = [mfma.zero_value] * N_ACCUMS

        for k in range_constexpr(K_ITERS):
            if const_expr(A_NARROW and not padded):
                # B6 BM192: clamp-wave combined G2S per 96-row A region.
                a_g2s_narrow.load(a_lds0, A0_off + k * KSTEP)
                a_g2s_narrow.load(a_lds1, A1_off + k * KSTEP)
            else:
                a_g2s.load(a_lds0, A0_off + k * KSTEP)
                a_g2s.load(a_lds1, A1_off + k * KSTEP)
            if const_expr(B_COMB):
                b_g2s.load(b_lds0, B0_off + k * KSTEP)
                b_g2s.load(b_lds1, B1_off + k * KSTEP)
            elif const_expr(B_NARROW):
                # BN64 (B4): clamped-wave combined G2S fills b_lds0 + LDS-adjacent
                # b_lds1 with the BLOCK_N B rows (all waves issue; >= NW_B_ACTIVE
                # redundantly reload the last region -> in-bounds, idempotent).
                b_g2s_narrow.load(b_lds0, B0_off + k * KSTEP)
            else:
                # BN128: one 128-row G2S over the full BLOCK_N rows fills both
                # adjacent half-regions (b_lds0 rows 0..63, b_lds1 rows 64..127).
                b_g2s_full.load(b_lds0, B0_off + k * KSTEP)
            if const_expr(padded):
                wait_barrier_lgkm()  # 2-hop ds_write completes on lgkm, not vmcnt
            else:
                wait_barrier(0)

            a0 = a_s2r.load(a_lds0)
            a1 = a_s2r.load(a_lds1)
            b0 = b_s2r.load(b_lds0)
            b1 = b_s2r.load(b_lds1)
            # per-sub-block scales (K128 index N_SUB*k + s)
            sa0 = [sa_s2r.load(sa_base0, N_SUB * k + s) for s in range_constexpr(N_SUB)]
            sa1 = [sa_s2r.load(sa_base1, N_SUB * k + s) for s in range_constexpr(N_SUB)]
            if const_expr(B_COMB):
                sb_all = [sb_s2r.load(sb_base0, N_SUB * k + s) for s in range_constexpr(N_SUB)]
                sb0 = [sb_all[s][0:2] for s in range_constexpr(N_SUB)]
                sb1 = [sb_all[s][2:4] for s in range_constexpr(N_SUB)]
            else:
                pairs = [sb_s2r.load_halves(sb_base0, LDS_BLOCK_N, N_SUB * k + s) for s in range_constexpr(N_SUB)]
                sb0 = [pairs[s][0] for s in range_constexpr(N_SUB)]
                sb1 = [pairs[s][1] for s in range_constexpr(N_SUB)]

            c00 = mfma.call_subs(a0, b0, c00, sa0, sb0, N_SUB)
            c01 = mfma.call_subs(a0, b1, c01, sa0, sb1, N_SUB)
            c10 = mfma.call_subs(a1, b0, c10, sa1, sb0, N_SUB)
            c11 = mfma.call_subs(a1, b1, c11, sa1, sb1, N_SUB)
            rocdl.s_barrier()

        base_row = block_m * BLOCK_M + wave_m_offset
        base_col = block_n * BLOCK_N + wave_n_offset
        store_c.store(c00, base_row + 0, base_col + 0)
        store_c.store(c01, base_row + 0, base_col + LDS_BLOCK_N)
        store_c.store(c10, base_row + LDS_BLOCK_M, base_col + 0)
        store_c.store(c11, base_row + LDS_BLOCK_M, base_col + LDS_BLOCK_N)

    @flyc.kernel(known_block_size=[512, 1, 1])
    def kernel_gemm_rt(
        A: fx.Tensor,
        B_T: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        c_m: fx.Int32,
        c_n: fx.Int32,
    ):
        # Runtime (NON-unrolled) K-loop carrying the 32 accumulators as iter_args,
        # so sched_group_barrier is emitted ONCE (no unroll explosion). Single
        # buffer + sched cadence: the LLVM scheduler interleaves S2R/G2S among the
        # MFMAs at the fp4 throughput rate (native LLIR-scheduler mechanism).
        F8_IR_t = fx.Float8E4M3FN.ir_type
        n_blocks = ceildiv(c_n, BLOCK_N)
        lds = fx.SharedAllocator().allocate(SharedStorageFp4).peek()
        a_lds0, a_lds1, b_lds0, b_lds1 = lds.A_lds0, lds.A_lds1, lds.B_lds0, lds.B_lds1

        lane_id = fx.thread_idx.x % 64
        wave_id = fx.thread_idx.x // 64
        wave_m = wave_id // 4
        wave_n = wave_id % 4
        block_m, block_n = grouped_xcd_pid(fx.block_idx.x, c_m, c_n, BLOCK_M, BLOCK_N,
                                           group_m=group_m, num_xcds=num_xcds, group_n=group_n)

        gA = make_fp8_buffer_tensor(A, F8_IR_t)
        gB = make_fp8_buffer_tensor(B_T, F8_IR_t)
        a_div = fx.logical_divide(gA, fx.make_layout(1, 1))
        b_div = fx.logical_divide(gB, fx.make_layout(1, 1))
        gl_off_a = fp4_g2s_offsets(lane_id, wave_id, K, N_LDS_STEPS_A, BPR)
        gl_off_b = fp4_g2s_offsets(lane_id, wave_id, K, N_LDS_STEPS_B, BPR)

        mfma = MfmaScaleFp4(N_TILES_A, N_TILES_B)
        a_g2s = G2SLoader(a_div, gl_off_a, N_LDS_STEPS_A, F8_IR_t, wave_id)
        b_g2s = G2SLoader(b_div, gl_off_b, N_LDS_STEPS_B, F8_IR_t, wave_id)
        a_s2r = S2RLoaderFp4(wave_m, N_TILES_A, N_SUB, BPR, LDS_ROW_STRIDE)
        b_s2r = S2RLoaderFp4(wave_n, N_TILES_B, N_SUB, BPR, LDS_ROW_STRIDE)
        # A-scale num_records must cover the rows the kernel addresses =
        # ceil(c_m/BLOCK_M)*BLOCK_M (the padded extent). ScaleS2R floors
        # dim//group_span, so pass a group_span-ceil'd dim. For aligned M
        # (16*SA_TILES | c_m, true for all BM256 production shapes) this == c_m
        # (byte-identical); only BM192 (16*3=48 ∤ 4096) pads, covering the edge
        # M-tile's scale group (else valid rows 4080-4095 clamp to scale 0 -> SNR 24).
        _sa_q = 16 * SA_TILES
        sa_s2r = ScaleS2R(A_scale, ((c_m + _sa_q - 1) // _sa_q) * _sa_q, K, SA_TILES)
        sb_s2r = ScaleBComb(B_scale, c_n, K)
        store_c = (StoreCAtomic if split_k > 1 else StoreCPlain)(C, c_m, c_n, mfma.idx, N_TILES_A, N_TILES_B)

        wave_m_offset = wave_m * (N_TILES_A * 16)
        wave_n_offset = wave_n * (N_TILES_B * 16)
        sa_b0 = fx.Int32(block_m * BLOCK_M + wave_m_offset)
        sa_b1 = sa_b0 + fx.Int32(LDS_BLOCK_M)
        sb_b0 = fx.Int32(block_n * BLOCK_N + wave_n_offset)
        A0_off = block_m * BLOCK_M * K2
        A1_off = (block_m * BLOCK_M + LDS_BLOCK_M) * K2
        B0_off = block_n * BLOCK_N * K2
        B1_off = (block_n * BLOCK_N + LDS_BLOCK_N) * K2

        init_args = ([mfma.zero_value] * N_ACCUMS) * 4  # c00,c01,c10,c11 flattened
        DSRD = 2 * (N_TILES_A + N_TILES_B)
        loop_results = init_args
        for k, args in range(0, K_ITERS, 1, init=init_args):
            ki = arith.index_cast(T.i32, k)
            c00 = list(args[0 * N_ACCUMS:1 * N_ACCUMS])
            c01 = list(args[1 * N_ACCUMS:2 * N_ACCUMS])
            c10 = list(args[2 * N_ACCUMS:3 * N_ACCUMS])
            c11 = list(args[3 * N_ACCUMS:4 * N_ACCUMS])
            a_g2s.load(a_lds0, A0_off + ki * KSTEP)
            a_g2s.load(a_lds1, A1_off + ki * KSTEP)
            b_g2s.load(b_lds0, B0_off + ki * KSTEP)
            b_g2s.load(b_lds1, B1_off + ki * KSTEP)
            wait_barrier(0)
            a0 = a_s2r.load(a_lds0)
            a1 = a_s2r.load(a_lds1)
            b0 = b_s2r.load(b_lds0)
            b1 = b_s2r.load(b_lds1)
            sa0 = sa_s2r.load(sa_b0, ki)
            sa1 = sa_s2r.load(sa_b1, ki)
            sb_all = sb_s2r.load(sb_b0, ki)
            sb0, sb1 = sb_all[0:2], sb_all[2:4]
            if const_expr(iglp):
                rocdl.iglp_opt(0)  # native AMDGPU iglp: GEMM MFMA/memory pipeline (= LLIR-sched effect)
            quads = [(c00, a0, b0, sa0, sb0), (c01, a0, b1, sa0, sb1),
                     (c10, a1, b0, sa1, sb0), (c11, a1, b1, sa1, sb1)]
            il_mma(mfma, quads, [], False, N_TILES_A, N_TILES_B,
                   sched_mm=(1 if sched else 0), sched_dsrd_n=DSRD)
            rocdl.s_barrier()
            loop_results = yield (c00 + c01 + c10 + c11)

        c00 = loop_results[0 * N_ACCUMS:1 * N_ACCUMS]
        c01 = loop_results[1 * N_ACCUMS:2 * N_ACCUMS]
        c10 = loop_results[2 * N_ACCUMS:3 * N_ACCUMS]
        c11 = loop_results[3 * N_ACCUMS:4 * N_ACCUMS]
        base_row = block_m * BLOCK_M + wave_m_offset
        base_col = block_n * BLOCK_N + wave_n_offset
        store_c.store(c00, base_row + 0, base_col + 0)
        store_c.store(c01, base_row + 0, base_col + LDS_BLOCK_N)
        store_c.store(c10, base_row + LDS_BLOCK_M, base_col + 0)
        store_c.store(c11, base_row + LDS_BLOCK_M, base_col + LDS_BLOCK_N)

    @fx.struct
    class SharedStorageFp4Pipe:
        A_lds_cur_0: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        A_lds_cur_1: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        A_lds_next_0: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        A_lds_next_1: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        B_lds_cur_0: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
        B_lds_cur_1: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
        B_lds_next_0: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
        B_lds_next_1: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]

    # B8 (round-45 r8_2): 3-stage LDS ring -> producer runs 2 K-iters ahead so the
    # k+2 G2S targets a FRESH stage buffer (never the just-read one), making the
    # per-sub-burst read-before-overwrite barriers redundant (removed in r8_3).
    # 12 buffers (3 stages x A/B x 2 halves) = 96KB of 160KB; occupancy-free (kernel
    # is VGPR=228-bound -> already 1 WG/CU, +32KB LDS stays 1 WG/CU). See
    # rounds/round-44/B8_DEEPER_RING_DESIGN.md.
    @fx.struct
    class SharedStorageFp4Pipe3:
        A_s0_0: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        A_s0_1: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        A_s1_0: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        A_s1_1: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        A_s2_0: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        A_s2_1: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        B_s0_0: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
        B_s0_1: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
        B_s1_0: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
        B_s1_1: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
        B_s2_0: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
        B_s2_1: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]

    @flyc.kernel(known_block_size=[512, 1, 1])
    def kernel_gemm_pipe(
        A: fx.Tensor,
        B_T: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        c_m: fx.Int32,
        c_n: fx.Int32,
    ):
        # fp4 software pipeline: mirror of mxfp8_gemm_8wave's double-buffered
        # cur/next staging with interleaved s_barriers + s_setprio + 1-deep scale
        # prefetch. Per K-iter contracts BLOCK_K fp4 == N_SUB 128-K MFMA sub-blocks.
        F8_IR_t = fx.Float8E4M3FN.ir_type

        lds = fx.SharedAllocator().allocate(SharedStorageFp4Pipe).peek()
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
        if const_expr(split_k > 1):
            split = fx.block_idx.x % split_k
            tile_pid = fx.block_idx.x // split_k
            KO = split * KI                    # runtime K-iter offset for this split
            ko_bytes = KO * KSTEP              # gmem byte offset along the row
        else:
            tile_pid = fx.block_idx.x
            KO = 0                              # python int -> scale kiter stays compile-time
            ko_bytes = 0
        block_m, block_n = grouped_xcd_pid(tile_pid, c_m, c_n, BLOCK_M, BLOCK_N,
                                           group_m=group_m, num_xcds=num_xcds, group_n=group_n)

        A0_off = block_m * BLOCK_M * K2 + ko_bytes
        A1_off = (block_m * BLOCK_M + LDS_BLOCK_M) * K2 + ko_bytes
        B0_off = block_n * BLOCK_N * K2 + ko_bytes
        B1_off = (block_n * BLOCK_N + LDS_BLOCK_N) * K2 + ko_bytes

        gA = make_fp8_buffer_tensor(A, F8_IR_t)
        gB = make_fp8_buffer_tensor(B_T, F8_IR_t)
        a_div = fx.logical_divide(gA, fx.make_layout(1, 1))
        b_div = fx.logical_divide(gB, fx.make_layout(1, 1))

        mfma = MfmaScaleFp4(N_TILES_A, N_TILES_B, asm=asm_mfma, asm_se=asm_se)
        if const_expr(padded):
            a_g2s = PaddedG2SLoader(A, c_m, K, lane_id, wave_id, N_LDS_STEPS_A, BPR, LDS_ROW_STRIDE)
            b_g2s = PaddedG2SLoader(B_T, c_n, K, lane_id, wave_id, N_LDS_STEPS_B, BPR, LDS_ROW_STRIDE)
        else:
            gl_off_a = fp4_g2s_offsets(lane_id, wave_id, K, N_LDS_STEPS_A, BPR)
            gl_off_b = fp4_g2s_offsets(lane_id, wave_id, K, N_LDS_STEPS_B, BPR)
            gl_off_b_full = fp4_g2s_offsets(lane_id, wave_id, K, N_LDS_STEPS_B_FULL, BPR)
            a_g2s = G2SLoader(a_div, gl_off_a, N_LDS_STEPS_A, F8_IR_t, wave_id)
            b_g2s = G2SLoader(b_div, gl_off_b, N_LDS_STEPS_B, F8_IR_t, wave_id)
            # BN128: combined-128 B G2S (one 128-row G2S fills adjacent b0+b1 halves).
            b_g2s_full = G2SLoader(b_div, gl_off_b_full, N_LDS_STEPS_B_FULL, F8_IR_t, wave_id)
            # B6 BM192 (r6_3): clamp-wave narrow-A G2S. LDS_BLOCK_M=96 < 128 step ->
            # N_LDS_STEPS_A==0 (regular a_g2s loads nothing). Mirror staged (r6_2):
            # each 96-row A region filled by one combined G2S step where the wave
            # index is clamped to [0, NW_A_ACTIVE-1]; waves >= NW_A_ACTIVE redundantly
            # reload the last 16-row region (same gmem src + LDS dst -> idempotent,
            # in-bounds, det-safe). All 8 waves issue unconditionally (no dynamic
            # per-wave if -> avoids FlyDSL stateful-object-in-branch TypeError).
            if const_expr(A_NARROW):
                _wid_a = fx.Int32(wave_id)
                eff_wave_a = (_wid_a < fx.Int32(NW_A_ACTIVE)).select(_wid_a, fx.Int32(max(NW_A_ACTIVE - 1, 0)))
                gl_off_a_narrow = fp4_g2s_offsets(lane_id, eff_wave_a, K, 1, BPR)
                a_g2s_narrow = G2SLoader(a_div, gl_off_a_narrow, 1, F8_IR_t, eff_wave_a)
        a_s2r = S2RLoaderFp4(wave_m, N_TILES_A, N_SUB, BPR, LDS_ROW_STRIDE, pad=(not asm_mfma) and frag_pad)
        b_s2r = S2RLoaderFp4(wave_n, N_TILES_B, N_SUB, BPR, LDS_ROW_STRIDE, pad=(not asm_mfma) and frag_pad)
        # A-scale num_records must cover the rows the kernel addresses =
        # ceil(c_m/BLOCK_M)*BLOCK_M (the padded extent). ScaleS2R floors
        # dim//group_span, so pass a group_span-ceil'd dim. For aligned M
        # (16*SA_TILES | c_m, true for all BM256 production shapes) this == c_m
        # (byte-identical); only BM192 (16*3=48 ∤ 4096) pads, covering the edge
        # M-tile's scale group (else valid rows 4080-4095 clamp to scale 0 -> SNR 24).
        _sa_q = 16 * SA_TILES
        sa_s2r = ScaleS2R(A_scale, ((c_m + _sa_q - 1) // _sa_q) * _sa_q, K, SA_TILES)
        sb_s2r = ScaleBComb(B_scale, c_n, K) if B_COMB else ScaleBRegion(B_scale, c_n, K, N_TILES_B)
        store_c = (StoreCAtomic if split_k > 1 else StoreCPlain)(C, c_m, c_n, mfma.idx, N_TILES_A, N_TILES_B)

        wave_m_offset = wave_m * (N_TILES_A * 16)
        wave_n_offset = wave_n * (N_TILES_B * 16)
        sa_base0 = fx.Int32(block_m * BLOCK_M + wave_m_offset)
        sa_base1 = sa_base0 + fx.Int32(LDS_BLOCK_M)
        sb_base0 = fx.Int32(block_n * BLOCK_N + wave_n_offset)

        # B6 r6_3: A G2S dispatch. BM192 (A_NARROW) -> clamp-wave narrow loader;
        # BM256 -> regular a_g2s (byte-identical). const_expr-gated so only the
        # selected path is traced. Pure side-effect (no captured-var leakage).
        def a_load(dst, off):
            if const_expr(A_NARROW and not padded):
                a_g2s_narrow.load(dst, off)
            else:
                a_g2s.load(dst, off)

        # Per-sub-block scale loaders (K128 index = N_SUB*kiter + s). _sb uses the
        # branch-free `load_halves` so BLOCK_N=128 (per-region) and BLOCK_N=256
        # (combined dwordx4) share one code path; BN256 packing is unchanged.
        def _sa(base, kiter):
            return [sa_s2r.load(base, N_SUB * kiter + s) for s in range_constexpr(N_SUB)]

        def _sb(base, kiter):
            pairs = [sb_s2r.load_halves(base, LDS_BLOCK_N, N_SUB * kiter + s) for s in range_constexpr(N_SUB)]
            return [pairs[s][0] for s in range_constexpr(N_SUB)], [pairs[s][1] for s in range_constexpr(N_SUB)]

        c00_frag = [mfma.zero_value] * N_ACCUMS
        c01_frag = [mfma.zero_value] * N_ACCUMS
        c10_frag = [mfma.zero_value] * N_ACCUMS
        c11_frag = [mfma.zero_value] * N_ACCUMS

        if const_expr(B_COMB):
            b_g2s.load(b_cur0, B0_off + 0 * KSTEP)
        else:
            b_g2s_full.load(b_cur0, B0_off + 0 * KSTEP)  # 128-row combined -> b_cur0+b_cur1
        a_load(a_cur0, A0_off + 0 * KSTEP)
        if const_expr(B_COMB):
            b_g2s.load(b_cur1, B1_off + 0 * KSTEP)
        a_load(a_cur1, A1_off + 0 * KSTEP)

        if wave_m == 1:
            rocdl.s_barrier()

        if const_expr(B_COMB):
            wait_barrier(N_LDS_STEPS_A + N_LDS_STEPS_B)
        else:
            wait_barrier(0)  # BN128: conservative prologue drain (one-time, correctness-first)

        if const_expr(B_COMB):
            b_g2s.load(b_next0, B0_off + 1 * KSTEP)
        else:
            b_g2s_full.load(b_next0, B0_off + 1 * KSTEP)  # 128-row combined -> b_next0+b_next1
        a_load(a_next0, A0_off + 1 * KSTEP)
        if const_expr(B_COMB):
            b_g2s.load(b_next1, B1_off + 1 * KSTEP)

        if const_expr(B_COMB):
            wait_barrier(N_LDS_STEPS_A + 2 * N_LDS_STEPS_B)
        else:
            wait_barrier(0)  # BN128: conservative prologue drain

        sa0 = _sa(sa_base0, KO + 0)
        sa1 = _sa(sa_base1, KO + 0)
        sb0, sb1 = _sb(sb_base0, KO + 0)

        for k in range_constexpr(KI - 2):
            ua = False if k == 0 else asm_mfma  # k==0 intrinsic zero-init, k>0 opaque asm
            sa0n = _sa(sa_base0, KO + k + 1)

            b0_frag = b_s2r.load(b_cur0)
            a0_frag = a_s2r.load(a_cur0)
            a_load(a_next1, A1_off + (k + 1) * KSTEP)
            rocdl.s_barrier()

            rocdl.s_setprio(1)
            c00_frag = mfma.call_subs(a0_frag, b0_frag, c00_frag, sa0, sb0, N_SUB, ua)
            rocdl.s_setprio(0)
            rocdl.s_barrier()

            b1_frag = b_s2r.load(b_cur1)
            if const_expr(B_COMB):
                b_g2s.load(b_cur0, B0_off + (k + 2) * KSTEP)
            sb0n, sb1n = _sb(sb_base0, KO + k + 1)
            rocdl.s_barrier()

            rocdl.s_setprio(1)
            c01_frag = mfma.call_subs(a0_frag, b1_frag, c01_frag, sa0, sb1, N_SUB, ua)
            rocdl.s_setprio(0)
            rocdl.s_barrier()

            a1_frag = a_s2r.load(a_cur1)
            a_load(a_cur0, A0_off + (k + 2) * KSTEP)
            sa1n = _sa(sa_base1, KO + k + 1)
            rocdl.s_barrier()

            rocdl.s_setprio(1)
            c10_frag = mfma.call_subs(a1_frag, b0_frag, c10_frag, sa1, sb0, N_SUB, ua)
            rocdl.s_setprio(0)
            rocdl.s_barrier()

            if const_expr(B_COMB):
                b_g2s.load(b_cur1, B1_off + (k + 2) * KSTEP)
                wait_barrier(2 * N_LDS_STEPS_A + N_LDS_STEPS_B)
            else:
                # BN128 combined k+2 (b_cur0+b_cur1). WAR-safe: b0/b1 reads above are
                # consumed by the c00/c01/c10 mmas before this overwrite.
                b_g2s_full.load(b_cur0, B0_off + (k + 2) * KSTEP)
                # r_k7: wait(3) raced (r_k6); try the tightest overlap-allowing drain = 1
                # outstanding (combined-spill writes nearly fully drained, but A-refills can
                # overlap). det0-gated: if this races too, the merged-spill needs vmcnt(0)
                # and pipe-BN128 perf is infeasible (conservative ≈ staged = loses).
                wait_barrier(1)

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

        # Step k = K_ITERS - 2 (prefetch last iter)
        k = KI - 2
        sa0n = _sa(sa_base0, KO + KI - 1)
        sa1n = _sa(sa_base1, KO + KI - 1)
        sb0n, sb1n = _sb(sb_base0, KO + KI - 1)

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
        a_load(a_next1, A1_off + (KI - 1) * KSTEP)
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

        # Step k = K_ITERS - 1
        k = KI - 1
        a0_frag = a_s2r.load(a_cur0)
        wait_barrier(0)

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
        rocdl.s_barrier()

        rocdl.s_setprio(1)
        c10_frag = mfma.call_subs(a1_frag, b0_frag, c10_frag, sa1, sb0, N_SUB, asm_mfma)
        c11_frag = mfma.call_subs(a1_frag, b1_frag, c11_frag, sa1, sb1, N_SUB, asm_mfma)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        base_row = block_m * BLOCK_M + wave_m_offset
        base_col = block_n * BLOCK_N + wave_n_offset
        store_c.store(c00_frag, base_row + 0, base_col + 0)
        store_c.store(c01_frag, base_row + 0, base_col + LDS_BLOCK_N)
        store_c.store(c10_frag, base_row + LDS_BLOCK_M, base_col + 0)
        store_c.store(c11_frag, base_row + LDS_BLOCK_M, base_col + LDS_BLOCK_N)

    @flyc.kernel(known_block_size=[512, 1, 1])
    def kernel_gemm_pipe3(
        A: fx.Tensor,
        B_T: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        c_m: fx.Int32,
        c_n: fx.Int32,
    ):
        # B8 r8_2: 3-stage LDS ring (cur/next/next2). Bulk-only (BN256/BM256/B_COMB,
        # split_k==1, no A_NARROW/padded). Producer runs 2 K-iters ahead -> the k+2
        # G2S targets a FRESH stage[(k+2)%3] (never the just-read stage[k%3]) so the
        # in-place read-before-overwrite hazard that forced pipe's ~8 barrier/K-iter
        # is GONE. r8_2 (this) keeps the per-sub-burst s_barrier (correctness-first,
        # isolate rotation correctness) + one full vmcnt-drain wait_barrier(0)/iter;
        # r8_3 removes the now-redundant s_barriers to coarsen toward aiter's ~1/iter.
        F8_IR_t = fx.Float8E4M3FN.ir_type

        lds = fx.SharedAllocator().allocate(SharedStorageFp4Pipe3).peek()
        A_lds = [[lds.A_s0_0, lds.A_s0_1], [lds.A_s1_0, lds.A_s1_1], [lds.A_s2_0, lds.A_s2_1]]
        B_lds = [[lds.B_s0_0, lds.B_s0_1], [lds.B_s1_0, lds.B_s1_1], [lds.B_s2_0, lds.B_s2_1]]

        lane_id = fx.thread_idx.x % 64
        wave_id = fx.thread_idx.x // 64
        wave_m = wave_id // 4
        wave_n = wave_id % 4
        block_m, block_n = grouped_xcd_pid(fx.block_idx.x, c_m, c_n, BLOCK_M, BLOCK_N,
                                           group_m=group_m, num_xcds=num_xcds, group_n=group_n)

        A0_off = block_m * BLOCK_M * K2
        A1_off = (block_m * BLOCK_M + LDS_BLOCK_M) * K2
        B0_off = block_n * BLOCK_N * K2
        B1_off = (block_n * BLOCK_N + LDS_BLOCK_N) * K2

        gA = make_fp8_buffer_tensor(A, F8_IR_t)
        gB = make_fp8_buffer_tensor(B_T, F8_IR_t)
        a_div = fx.logical_divide(gA, fx.make_layout(1, 1))
        b_div = fx.logical_divide(gB, fx.make_layout(1, 1))

        mfma = MfmaScaleFp4(N_TILES_A, N_TILES_B, asm=asm_mfma, asm_se=asm_se)
        gl_off_a = fp4_g2s_offsets(lane_id, wave_id, K, N_LDS_STEPS_A, BPR)
        gl_off_b = fp4_g2s_offsets(lane_id, wave_id, K, N_LDS_STEPS_B, BPR)
        a_g2s = G2SLoader(a_div, gl_off_a, N_LDS_STEPS_A, F8_IR_t, wave_id)
        b_g2s = G2SLoader(b_div, gl_off_b, N_LDS_STEPS_B, F8_IR_t, wave_id)
        a_s2r = S2RLoaderFp4(wave_m, N_TILES_A, N_SUB, BPR, LDS_ROW_STRIDE, pad=(not asm_mfma) and frag_pad)
        b_s2r = S2RLoaderFp4(wave_n, N_TILES_B, N_SUB, BPR, LDS_ROW_STRIDE, pad=(not asm_mfma) and frag_pad)
        _sa_q = 16 * SA_TILES
        sa_s2r = ScaleS2R(A_scale, ((c_m + _sa_q - 1) // _sa_q) * _sa_q, K, SA_TILES)
        sb_s2r = ScaleBComb(B_scale, c_n, K)
        store_c = StoreCPlain(C, c_m, c_n, mfma.idx, N_TILES_A, N_TILES_B)

        wave_m_offset = wave_m * (N_TILES_A * 16)
        wave_n_offset = wave_n * (N_TILES_B * 16)
        sa_base0 = fx.Int32(block_m * BLOCK_M + wave_m_offset)
        sa_base1 = sa_base0 + fx.Int32(LDS_BLOCK_M)
        sb_base0 = fx.Int32(block_n * BLOCK_N + wave_n_offset)

        def _sa(base, kiter):
            return [sa_s2r.load(base, N_SUB * kiter + s) for s in range_constexpr(N_SUB)]

        def _sb(base, kiter):
            pairs = [sb_s2r.load_halves(base, LDS_BLOCK_N, N_SUB * kiter + s) for s in range_constexpr(N_SUB)]
            return [pairs[s][0] for s in range_constexpr(N_SUB)], [pairs[s][1] for s in range_constexpr(N_SUB)]

        c00_frag = [mfma.zero_value] * N_ACCUMS
        c01_frag = [mfma.zero_value] * N_ACCUMS
        c10_frag = [mfma.zero_value] * N_ACCUMS
        c11_frag = [mfma.zero_value] * N_ACCUMS

        # Prologue: prime stage 0 (k=0) and stage 1 (k=1), all 4 buffers each.
        b_g2s.load(B_lds[0][0], B0_off + 0 * KSTEP)
        a_g2s.load(A_lds[0][0], A0_off + 0 * KSTEP)
        b_g2s.load(B_lds[0][1], B1_off + 0 * KSTEP)
        a_g2s.load(A_lds[0][1], A1_off + 0 * KSTEP)
        if wave_m == 1:
            rocdl.s_barrier()
        wait_barrier(0)
        if const_expr(KI > 1):
            b_g2s.load(B_lds[1][0], B0_off + 1 * KSTEP)
            a_g2s.load(A_lds[1][0], A0_off + 1 * KSTEP)
            b_g2s.load(B_lds[1][1], B1_off + 1 * KSTEP)
            a_g2s.load(A_lds[1][1], A1_off + 1 * KSTEP)
        wait_barrier(0)

        sa0 = _sa(sa_base0, 0)
        sa1 = _sa(sa_base1, 0)
        sb0, sb1 = _sb(sb_base0, 0)

        for k in range_constexpr(KI):
            cur = k % 3
            n2 = (k + 2) % 3
            ua = False if k == 0 else asm_mfma
            if const_expr(k + 1 < KI):
                sa0n = _sa(sa_base0, k + 1)

            b0_frag = b_s2r.load(B_lds[cur][0])
            a0_frag = a_s2r.load(A_lds[cur][0])
            if const_expr(k + 2 < KI):
                a_g2s.load(A_lds[n2][1], A1_off + (k + 2) * KSTEP)
            rocdl.s_barrier()
            rocdl.s_setprio(1)
            c00_frag = mfma.call_subs(a0_frag, b0_frag, c00_frag, sa0, sb0, N_SUB, ua)
            rocdl.s_setprio(0)
            rocdl.s_barrier()

            b1_frag = b_s2r.load(B_lds[cur][1])
            if const_expr(k + 2 < KI):
                b_g2s.load(B_lds[n2][0], B0_off + (k + 2) * KSTEP)
            if const_expr(k + 1 < KI):
                sb0n, sb1n = _sb(sb_base0, k + 1)
            rocdl.s_barrier()
            rocdl.s_setprio(1)
            c01_frag = mfma.call_subs(a0_frag, b1_frag, c01_frag, sa0, sb1, N_SUB, ua)
            rocdl.s_setprio(0)
            rocdl.s_barrier()

            a1_frag = a_s2r.load(A_lds[cur][1])
            if const_expr(k + 2 < KI):
                a_g2s.load(A_lds[n2][0], A0_off + (k + 2) * KSTEP)
            if const_expr(k + 1 < KI):
                sa1n = _sa(sa_base1, k + 1)
            rocdl.s_barrier()
            rocdl.s_setprio(1)
            c10_frag = mfma.call_subs(a1_frag, b0_frag, c10_frag, sa1, sb0, N_SUB, ua)
            rocdl.s_setprio(0)
            rocdl.s_barrier()

            if const_expr(k + 2 < KI):
                b_g2s.load(B_lds[n2][1], B1_off + (k + 2) * KSTEP)
            wait_barrier(0)
            rocdl.s_setprio(1)
            c11_frag = mfma.call_subs(a1_frag, b1_frag, c11_frag, sa1, sb1, N_SUB, ua)
            rocdl.s_setprio(0)
            rocdl.s_barrier()

            if const_expr(k + 1 < KI):
                sa0, sa1 = sa0n, sa1n
                sb0, sb1 = sb0n, sb1n

        base_row = block_m * BLOCK_M + wave_m_offset
        base_col = block_n * BLOCK_N + wave_n_offset
        store_c.store(c00_frag, base_row + 0, base_col + 0)
        store_c.store(c01_frag, base_row + 0, base_col + LDS_BLOCK_N)
        store_c.store(c10_frag, base_row + LDS_BLOCK_M, base_col + 0)
        store_c.store(c11_frag, base_row + LDS_BLOCK_M, base_col + LDS_BLOCK_N)

    @flyc.kernel(known_block_size=[512, 1, 1])
    def kernel_gemm_pipeb(
        A: fx.Tensor,
        B_T: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        c_m: fx.Int32,
        c_n: fx.Int32,
    ):
        # Like `pipe`, but the B fragments are register-prefetched one K-iter ahead
        # (read b_next -> b{0,1}_reg during this iter's MFMAs), so the B ds_read
        # latency hides under MFMA instead of stalling on lgkmcnt(0). A stays
        # just-in-time. +16 VGPR (B double-buffer) fits the 256 budget. WAR-safe:
        # prefetch reads b_next (iter k+1), G2S writes b_cur (iter k+2); disjoint.
        F8_IR_t = fx.Float8E4M3FN.ir_type
        lds = fx.SharedAllocator().allocate(SharedStorageFp4Pipe).peek()
        a_cur0 = lds.A_lds_cur_0; a_cur1 = lds.A_lds_cur_1
        a_next0 = lds.A_lds_next_0; a_next1 = lds.A_lds_next_1
        b_cur0 = lds.B_lds_cur_0; b_cur1 = lds.B_lds_cur_1
        b_next0 = lds.B_lds_next_0; b_next1 = lds.B_lds_next_1

        lane_id = fx.thread_idx.x % 64
        wave_id = fx.thread_idx.x // 64
        wave_m = wave_id // 4
        wave_n = wave_id % 4
        block_m, block_n = grouped_xcd_pid(fx.block_idx.x, c_m, c_n, BLOCK_M, BLOCK_N,
                                           group_m=group_m, num_xcds=num_xcds, group_n=group_n)
        A0_off = block_m * BLOCK_M * K2
        A1_off = (block_m * BLOCK_M + LDS_BLOCK_M) * K2
        B0_off = block_n * BLOCK_N * K2
        B1_off = (block_n * BLOCK_N + LDS_BLOCK_N) * K2

        gA = make_fp8_buffer_tensor(A, F8_IR_t)
        gB = make_fp8_buffer_tensor(B_T, F8_IR_t)
        a_div = fx.logical_divide(gA, fx.make_layout(1, 1))
        b_div = fx.logical_divide(gB, fx.make_layout(1, 1))

        mfma = MfmaScaleFp4(N_TILES_A, N_TILES_B)
        gl_off_a = fp4_g2s_offsets(lane_id, wave_id, K, N_LDS_STEPS_A, BPR)
        gl_off_b = fp4_g2s_offsets(lane_id, wave_id, K, N_LDS_STEPS_B, BPR)
        a_g2s = G2SLoader(a_div, gl_off_a, N_LDS_STEPS_A, F8_IR_t, wave_id)
        b_g2s = G2SLoader(b_div, gl_off_b, N_LDS_STEPS_B, F8_IR_t, wave_id)
        a_s2r = S2RLoaderFp4(wave_m, N_TILES_A, N_SUB, BPR, LDS_ROW_STRIDE, pad=frag_pad)
        b_s2r = S2RLoaderFp4(wave_n, N_TILES_B, N_SUB, BPR, LDS_ROW_STRIDE, pad=frag_pad)
        # A-scale num_records must cover the rows the kernel addresses =
        # ceil(c_m/BLOCK_M)*BLOCK_M (the padded extent). ScaleS2R floors
        # dim//group_span, so pass a group_span-ceil'd dim. For aligned M
        # (16*SA_TILES | c_m, true for all BM256 production shapes) this == c_m
        # (byte-identical); only BM192 (16*3=48 ∤ 4096) pads, covering the edge
        # M-tile's scale group (else valid rows 4080-4095 clamp to scale 0 -> SNR 24).
        _sa_q = 16 * SA_TILES
        sa_s2r = ScaleS2R(A_scale, ((c_m + _sa_q - 1) // _sa_q) * _sa_q, K, SA_TILES)
        sb_s2r = ScaleBComb(B_scale, c_n, K)
        store_c = StoreCPlain(C, c_m, c_n, mfma.idx, N_TILES_A, N_TILES_B)

        wave_m_offset = wave_m * (N_TILES_A * 16)
        wave_n_offset = wave_n * (N_TILES_B * 16)
        sa_base0 = fx.Int32(block_m * BLOCK_M + wave_m_offset)
        sa_base1 = sa_base0 + fx.Int32(LDS_BLOCK_M)
        sb_base0 = fx.Int32(block_n * BLOCK_N + wave_n_offset)

        def _sa(base, kiter):
            return [sa_s2r.load(base, N_SUB * kiter + s) for s in range_constexpr(N_SUB)]

        def _sb(base, kiter):
            alls = [sb_s2r.load(base, N_SUB * kiter + s) for s in range_constexpr(N_SUB)]
            return [alls[s][0:2] for s in range_constexpr(N_SUB)], [alls[s][2:4] for s in range_constexpr(N_SUB)]

        c00_frag = [mfma.zero_value] * N_ACCUMS
        c01_frag = [mfma.zero_value] * N_ACCUMS
        c10_frag = [mfma.zero_value] * N_ACCUMS
        c11_frag = [mfma.zero_value] * N_ACCUMS

        b_g2s.load(b_cur0, B0_off + 0 * KSTEP)
        a_g2s.load(a_cur0, A0_off + 0 * KSTEP)
        b_g2s.load(b_cur1, B1_off + 0 * KSTEP)
        a_g2s.load(a_cur1, A1_off + 0 * KSTEP)
        if wave_m == 1:
            rocdl.s_barrier()
        wait_barrier(N_LDS_STEPS_A + N_LDS_STEPS_B)

        b_g2s.load(b_next0, B0_off + 1 * KSTEP)
        a_g2s.load(a_next0, A0_off + 1 * KSTEP)
        b_g2s.load(b_next1, B1_off + 1 * KSTEP)
        wait_barrier(N_LDS_STEPS_A + 2 * N_LDS_STEPS_B)

        sa0 = _sa(sa_base0, 0); sa1 = _sa(sa_base1, 0)
        sb0, sb1 = _sb(sb_base0, 0)
        # Register-prefetch iter 0's B fragments (b_cur holds iter 0).
        b0_reg = b_s2r.load(b_cur0)
        b1_reg = b_s2r.load(b_cur1)

        for k in range_constexpr(KI - 2):
            sa0n = _sa(sa_base0, k + 1)
            # Prefetch iter k+1's B (b_next) into registers -- overlaps MFMAs below.
            b0_nxt = b_s2r.load(b_next0)
            b1_nxt = b_s2r.load(b_next1)
            a0_frag = a_s2r.load(a_cur0)
            a_g2s.load(a_next1, A1_off + (k + 1) * KSTEP)
            rocdl.s_barrier()

            rocdl.s_setprio(1)
            c00_frag = mfma.call_subs(a0_frag, b0_reg, c00_frag, sa0, sb0, N_SUB, False)
            rocdl.s_setprio(0)
            rocdl.s_barrier()

            b_g2s.load(b_cur0, B0_off + (k + 2) * KSTEP)
            sb0n, sb1n = _sb(sb_base0, k + 1)
            rocdl.s_barrier()

            rocdl.s_setprio(1)
            c01_frag = mfma.call_subs(a0_frag, b1_reg, c01_frag, sa0, sb1, N_SUB, False)
            rocdl.s_setprio(0)
            rocdl.s_barrier()

            a1_frag = a_s2r.load(a_cur1)
            a_g2s.load(a_cur0, A0_off + (k + 2) * KSTEP)
            sa1n = _sa(sa_base1, k + 1)
            rocdl.s_barrier()

            rocdl.s_setprio(1)
            c10_frag = mfma.call_subs(a1_frag, b0_reg, c10_frag, sa1, sb0, N_SUB, False)
            rocdl.s_setprio(0)
            rocdl.s_barrier()

            b_g2s.load(b_cur1, B1_off + (k + 2) * KSTEP)
            wait_barrier(2 * N_LDS_STEPS_A + N_LDS_STEPS_B)

            rocdl.s_setprio(1)
            c11_frag = mfma.call_subs(a1_frag, b1_reg, c11_frag, sa1, sb1, N_SUB, False)
            rocdl.s_setprio(0)
            rocdl.s_barrier()

            a_cur0, a_next0 = a_next0, a_cur0
            a_cur1, a_next1 = a_next1, a_cur1
            b_cur0, b_next0 = b_next0, b_cur0
            b_cur1, b_next1 = b_next1, b_cur1
            sa0, sa1 = sa0n, sa1n
            sb0, sb1 = sb0n, sb1n
            b0_reg, b1_reg = b0_nxt, b1_nxt

        # Epilogue step k = KI-2 (b0_reg/b1_reg hold iter KI-2 B; b_next holds KI-1)
        sa0n = _sa(sa_base0, KI - 1); sa1n = _sa(sa_base1, KI - 1)
        sb0n, sb1n = _sb(sb_base0, KI - 1)
        b0_nxt = b_s2r.load(b_next0)
        b1_nxt = b_s2r.load(b_next1)
        a0_frag = a_s2r.load(a_cur0)
        rocdl.s_barrier()
        rocdl.s_setprio(1)
        c00_frag = mfma.call_subs(a0_frag, b0_reg, c00_frag, sa0, sb0, N_SUB, False)
        rocdl.s_setprio(0)
        rocdl.s_barrier()
        rocdl.s_setprio(1)
        c01_frag = mfma.call_subs(a0_frag, b1_reg, c01_frag, sa0, sb1, N_SUB, False)
        rocdl.s_setprio(0)
        rocdl.s_barrier()
        a1_frag = a_s2r.load(a_cur1)
        rocdl.s_barrier()
        rocdl.s_setprio(1)
        c10_frag = mfma.call_subs(a1_frag, b0_reg, c10_frag, sa1, sb0, N_SUB, False)
        rocdl.s_setprio(0)
        rocdl.s_barrier()
        rocdl.s_setprio(1)
        c11_frag = mfma.call_subs(a1_frag, b1_reg, c11_frag, sa1, sb1, N_SUB, False)
        rocdl.s_setprio(0)
        rocdl.s_barrier()
        a_cur0, a_next0 = a_next0, a_cur0
        a_cur1, a_next1 = a_next1, a_cur1
        sa0, sa1 = sa0n, sa1n
        sb0, sb1 = sb0n, sb1n
        b0_reg, b1_reg = b0_nxt, b1_nxt

        # Epilogue step k = KI-1 (b0_reg/b1_reg hold iter KI-1 B)
        a0_frag = a_s2r.load(a_cur0)
        wait_barrier(0)
        rocdl.s_setprio(1)
        c00_frag = mfma.call_subs(a0_frag, b0_reg, c00_frag, sa0, sb0, N_SUB, False)
        rocdl.s_setprio(0)
        rocdl.s_barrier()
        c01_frag = mfma.call_subs(a0_frag, b1_reg, c01_frag, sa0, sb1, N_SUB, False)
        rocdl.s_barrier()
        a1_frag = a_s2r.load(a_cur1)
        rocdl.s_barrier()
        rocdl.s_setprio(1)
        c10_frag = mfma.call_subs(a1_frag, b0_reg, c10_frag, sa1, sb0, N_SUB, False)
        c11_frag = mfma.call_subs(a1_frag, b1_reg, c11_frag, sa1, sb1, N_SUB, False)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        base_row = block_m * BLOCK_M + wave_m_offset
        base_col = block_n * BLOCK_N + wave_n_offset
        store_c.store(c00_frag, base_row + 0, base_col + 0)
        store_c.store(c01_frag, base_row + 0, base_col + LDS_BLOCK_N)
        store_c.store(c10_frag, base_row + LDS_BLOCK_M, base_col + 0)
        store_c.store(c11_frag, base_row + LDS_BLOCK_M, base_col + LDS_BLOCK_N)

    @flyc.kernel(known_block_size=[512, 1, 1])
    def kernel_gemm_pipeh(
        A: fx.Tensor,
        B_T: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        c_m: fx.Int32,
        c_n: fx.Int32,
    ):
        # Hoisted variant of `pipe`: all 4 ds_reads issued at the top of each
        # K-iter (we have ~140 spare VGPR), then a single high-prio MFMA burst of
        # the 4 accumulator clusters while the next-iter G2S global loads stream
        # underneath. Mirrors the competitor's "all operands resident -> dense
        # MFMA burst with loads overlapped" structure, but in 8-wave layout. The
        # ds_read->same-LDS-buffer-overwrite WAR is covered by an explicit
        # wait_lgkm() (in `pipe` it was covered implicitly by the per-cluster MFMA
        # data dependency, which no longer sits between read and overwrite).
        F8_IR_t = fx.Float8E4M3FN.ir_type

        lds = fx.SharedAllocator().allocate(SharedStorageFp4Pipe).peek()
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
        block_m, block_n = grouped_xcd_pid(fx.block_idx.x, c_m, c_n, BLOCK_M, BLOCK_N,
                                           group_m=group_m, num_xcds=num_xcds, group_n=group_n)

        A0_off = block_m * BLOCK_M * K2
        A1_off = (block_m * BLOCK_M + LDS_BLOCK_M) * K2
        B0_off = block_n * BLOCK_N * K2
        B1_off = (block_n * BLOCK_N + LDS_BLOCK_N) * K2

        gA = make_fp8_buffer_tensor(A, F8_IR_t)
        gB = make_fp8_buffer_tensor(B_T, F8_IR_t)
        a_div = fx.logical_divide(gA, fx.make_layout(1, 1))
        b_div = fx.logical_divide(gB, fx.make_layout(1, 1))

        mfma = MfmaScaleFp4(N_TILES_A, N_TILES_B, asm=asm_mfma, asm_se=asm_se)
        gl_off_a = fp4_g2s_offsets(lane_id, wave_id, K, N_LDS_STEPS_A, BPR)
        gl_off_b = fp4_g2s_offsets(lane_id, wave_id, K, N_LDS_STEPS_B, BPR)
        a_g2s = G2SLoader(a_div, gl_off_a, N_LDS_STEPS_A, F8_IR_t, wave_id)
        b_g2s = G2SLoader(b_div, gl_off_b, N_LDS_STEPS_B, F8_IR_t, wave_id)
        a_s2r = S2RLoaderFp4(wave_m, N_TILES_A, N_SUB, BPR, LDS_ROW_STRIDE, pad=frag_pad)
        b_s2r = S2RLoaderFp4(wave_n, N_TILES_B, N_SUB, BPR, LDS_ROW_STRIDE, pad=frag_pad)
        # A-scale num_records must cover the rows the kernel addresses =
        # ceil(c_m/BLOCK_M)*BLOCK_M (the padded extent). ScaleS2R floors
        # dim//group_span, so pass a group_span-ceil'd dim. For aligned M
        # (16*SA_TILES | c_m, true for all BM256 production shapes) this == c_m
        # (byte-identical); only BM192 (16*3=48 ∤ 4096) pads, covering the edge
        # M-tile's scale group (else valid rows 4080-4095 clamp to scale 0 -> SNR 24).
        _sa_q = 16 * SA_TILES
        sa_s2r = ScaleS2R(A_scale, ((c_m + _sa_q - 1) // _sa_q) * _sa_q, K, SA_TILES)
        sb_s2r = ScaleBComb(B_scale, c_n, K)
        store_c = (StoreCAtomic if split_k > 1 else StoreCPlain)(C, c_m, c_n, mfma.idx, N_TILES_A, N_TILES_B)

        wave_m_offset = wave_m * (N_TILES_A * 16)
        wave_n_offset = wave_n * (N_TILES_B * 16)
        sa_base0 = fx.Int32(block_m * BLOCK_M + wave_m_offset)
        sa_base1 = sa_base0 + fx.Int32(LDS_BLOCK_M)
        sb_base0 = fx.Int32(block_n * BLOCK_N + wave_n_offset)

        def _sa(base, kiter):
            return [sa_s2r.load(base, N_SUB * kiter + s) for s in range_constexpr(N_SUB)]

        def _sb(base, kiter):
            alls = [sb_s2r.load(base, N_SUB * kiter + s) for s in range_constexpr(N_SUB)]
            return [alls[s][0:2] for s in range_constexpr(N_SUB)], [alls[s][2:4] for s in range_constexpr(N_SUB)]

        c00_frag = [mfma.zero_value] * N_ACCUMS
        c01_frag = [mfma.zero_value] * N_ACCUMS
        c10_frag = [mfma.zero_value] * N_ACCUMS
        c11_frag = [mfma.zero_value] * N_ACCUMS

        b_g2s.load(b_cur0, B0_off + 0 * KSTEP)
        a_g2s.load(a_cur0, A0_off + 0 * KSTEP)
        b_g2s.load(b_cur1, B1_off + 0 * KSTEP)
        a_g2s.load(a_cur1, A1_off + 0 * KSTEP)

        if wave_m == 1:
            rocdl.s_barrier()

        wait_barrier(N_LDS_STEPS_A + N_LDS_STEPS_B)

        b_g2s.load(b_next0, B0_off + 1 * KSTEP)
        a_g2s.load(a_next0, A0_off + 1 * KSTEP)
        b_g2s.load(b_next1, B1_off + 1 * KSTEP)
        a_g2s.load(a_next1, A1_off + 1 * KSTEP)

        wait_barrier(0)

        sa0 = _sa(sa_base0, 0)
        sa1 = _sa(sa_base1, 0)
        sb0, sb1 = _sb(sb_base0, 0)

        for k in range_constexpr(K_ITERS - 2):
            ua = False if k == 0 else asm_mfma
            # Hoist all four ds_reads (operands resident in VGPR before the burst).
            b0_frag = b_s2r.load(b_cur0)
            a0_frag = a_s2r.load(a_cur0)
            b1_frag = b_s2r.load(b_cur1)
            a1_frag = a_s2r.load(a_cur1)
            # Next-iter scale prefetch (vmem; not blocked by the lgkm wait below).
            sa0n = _sa(sa_base0, k + 1)
            sa1n = _sa(sa_base1, k + 1)
            sb0n, sb1n = _sb(sb_base0, k + 1)
            # Ensure the four ds_reads have landed before any G2S overwrites the
            # same LDS buffers (b_cur0/a_cur0/b_cur1/a_cur1 are reloaded with k+2).
            wait_lgkm()
            rocdl.s_barrier()  # all waves' ds_reads landed before G2S overwrites cur
            b_g2s.load(b_cur0, B0_off + (k + 2) * KSTEP)
            a_g2s.load(a_cur0, A0_off + (k + 2) * KSTEP)
            b_g2s.load(b_cur1, B1_off + (k + 2) * KSTEP)
            a_g2s.load(a_cur1, A1_off + (k + 2) * KSTEP)
            # Single dense MFMA burst; G2S global loads stream underneath it.
            rocdl.s_setprio(1)
            c00_frag = mfma.call_subs(a0_frag, b0_frag, c00_frag, sa0, sb0, N_SUB, ua)
            c01_frag = mfma.call_subs(a0_frag, b1_frag, c01_frag, sa0, sb1, N_SUB, ua)
            c10_frag = mfma.call_subs(a1_frag, b0_frag, c10_frag, sa1, sb0, N_SUB, ua)
            c11_frag = mfma.call_subs(a1_frag, b1_frag, c11_frag, sa1, sb1, N_SUB, ua)
            rocdl.s_setprio(0)
            wait_barrier(0)

            a_cur0, a_next0 = a_next0, a_cur0
            a_cur1, a_next1 = a_next1, a_cur1
            b_cur0, b_next0 = b_next0, b_cur0
            b_cur1, b_next1 = b_next1, b_cur1
            sa0, sa1 = sa0n, sa1n
            sb0, sb1 = sb0n, sb1n

        # Tail: last 2 K-iters (no further k+2 prefetch). cur buffers hold k, next
        # holds k+1 already; just drain.
        for kt in range_constexpr(2):
            k = K_ITERS - 2 + kt
            ua = asm_mfma
            b0_frag = b_s2r.load(b_cur0)
            a0_frag = a_s2r.load(a_cur0)
            b1_frag = b_s2r.load(b_cur1)
            a1_frag = a_s2r.load(a_cur1)
            if const_expr(kt == 0):
                sa0n = _sa(sa_base0, K_ITERS - 1)
                sa1n = _sa(sa_base1, K_ITERS - 1)
                sb0n, sb1n = _sb(sb_base0, K_ITERS - 1)
            wait_barrier(0)
            rocdl.s_setprio(1)
            c00_frag = mfma.call_subs(a0_frag, b0_frag, c00_frag, sa0, sb0, N_SUB, ua)
            c01_frag = mfma.call_subs(a0_frag, b1_frag, c01_frag, sa0, sb1, N_SUB, ua)
            c10_frag = mfma.call_subs(a1_frag, b0_frag, c10_frag, sa1, sb0, N_SUB, ua)
            c11_frag = mfma.call_subs(a1_frag, b1_frag, c11_frag, sa1, sb1, N_SUB, ua)
            rocdl.s_setprio(0)
            rocdl.s_barrier()
            if const_expr(kt == 0):
                a_cur0, a_next0 = a_next0, a_cur0
                a_cur1, a_next1 = a_next1, a_cur1
                b_cur0, b_next0 = b_next0, b_cur0
                b_cur1, b_next1 = b_next1, b_cur1
                sa0, sa1 = sa0n, sa1n
                sb0, sb1 = sb0n, sb1n

        base_row = block_m * BLOCK_M + wave_m_offset
        base_col = block_n * BLOCK_N + wave_n_offset
        store_c.store(c00_frag, base_row + 0, base_col + 0)
        store_c.store(c01_frag, base_row + 0, base_col + LDS_BLOCK_N)
        store_c.store(c10_frag, base_row + LDS_BLOCK_M, base_col + 0)
        store_c.store(c11_frag, base_row + LDS_BLOCK_M, base_col + LDS_BLOCK_N)

    @flyc.kernel(known_block_size=[512, 1, 1])
    def kernel_gemm_il(
        A: fx.Tensor,
        B_T: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        c_m: fx.Int32,
        c_n: fx.Int32,
    ):
        # Clean 8-wave interleaved: double-buffer, NO mid-MFMA barriers (one
        # wait_barrier per iter). prefetch G2S(k+1) load_one spread among the 32
        # inline-asm MFMAs -> async gmem->LDS overlaps MFMA; opaque asm blocks
        # LLVM re-clustering. No mid-barriers -> no AGPR copy storm (unlike pipe).
        F8_IR_t = fx.Float8E4M3FN.ir_type
        n_blocks = ceildiv(c_n, BLOCK_N)
        lds = fx.SharedAllocator().allocate(SharedStorageFp4Pipe).peek()
        a_cur0, a_cur1 = lds.A_lds_cur_0, lds.A_lds_cur_1
        a_nxt0, a_nxt1 = lds.A_lds_next_0, lds.A_lds_next_1
        b_cur0, b_cur1 = lds.B_lds_cur_0, lds.B_lds_cur_1
        b_nxt0, b_nxt1 = lds.B_lds_next_0, lds.B_lds_next_1

        lane_id = fx.thread_idx.x % 64
        wave_id = fx.thread_idx.x // 64
        wave_m = wave_id // 4
        wave_n = wave_id % 4
        block_m, block_n = grouped_xcd_pid(fx.block_idx.x, c_m, c_n, BLOCK_M, BLOCK_N,
                                           group_m=group_m, num_xcds=num_xcds, group_n=group_n)

        gA = make_fp8_buffer_tensor(A, F8_IR_t)
        gB = make_fp8_buffer_tensor(B_T, F8_IR_t)
        a_div = fx.logical_divide(gA, fx.make_layout(1, 1))
        b_div = fx.logical_divide(gB, fx.make_layout(1, 1))
        gl_off_a = fp4_g2s_offsets(lane_id, wave_id, K, N_LDS_STEPS_A, BPR)
        gl_off_b = fp4_g2s_offsets(lane_id, wave_id, K, N_LDS_STEPS_B, BPR)

        mfma = MfmaScaleFp4(N_TILES_A, N_TILES_B, asm=asm_mfma, asm_se=asm_se)
        a_g2s = G2SLoader(a_div, gl_off_a, N_LDS_STEPS_A, F8_IR_t, wave_id)
        b_g2s = G2SLoader(b_div, gl_off_b, N_LDS_STEPS_B, F8_IR_t, wave_id)
        a_s2r = S2RLoaderFp4(wave_m, N_TILES_A, N_SUB, BPR, LDS_ROW_STRIDE, pad=(not asm_mfma) and frag_pad)
        b_s2r = S2RLoaderFp4(wave_n, N_TILES_B, N_SUB, BPR, LDS_ROW_STRIDE, pad=(not asm_mfma) and frag_pad)
        sa_s2r = ScaleS2R(A_scale, c_m, K, N_TILES_A)
        sb_s2r = ScaleBComb(B_scale, c_n, K)
        store_c = (StoreCAtomic if split_k > 1 else StoreCPlain)(C, c_m, c_n, mfma.idx, N_TILES_A, N_TILES_B)

        wm_off = wave_m * (N_TILES_A * 16)
        wn_off = wave_n * (N_TILES_B * 16)
        sa_b0 = fx.Int32(block_m * BLOCK_M + wm_off)
        sa_b1 = sa_b0 + fx.Int32(LDS_BLOCK_M)
        sb_b0 = fx.Int32(block_n * BLOCK_N + wn_off)
        A0o = block_m * BLOCK_M * K2
        A1o = (block_m * BLOCK_M + LDS_BLOCK_M) * K2
        B0o = block_n * BLOCK_N * K2
        B1o = (block_n * BLOCK_N + LDS_BLOCK_N) * K2

        c00 = [mfma.zero_value] * N_ACCUMS
        c01 = [mfma.zero_value] * N_ACCUMS
        c10 = [mfma.zero_value] * N_ACCUMS
        c11 = [mfma.zero_value] * N_ACCUMS

        a_g2s.load(a_cur0, A0o + 0 * KSTEP)
        a_g2s.load(a_cur1, A1o + 0 * KSTEP)
        b_g2s.load(b_cur0, B0o + 0 * KSTEP)
        b_g2s.load(b_cur1, B1o + 0 * KSTEP)
        wait_barrier(0)

        n_mma = 4 * N_TILES_A * N_TILES_B  # 32
        n_pf = 2 * N_LDS_STEPS_A + 2 * N_LDS_STEPS_B
        per = max(1, n_mma // n_pf)

        for k in range_constexpr(K_ITERS - 1):
            a0 = a_s2r.load(a_cur0)
            a1 = a_s2r.load(a_cur1)
            b0 = b_s2r.load(b_cur0)
            b1 = b_s2r.load(b_cur1)
            sa0 = sa_s2r.load(sa_b0, k)
            sa1 = sa_s2r.load(sa_b1, k)
            sb_all = sb_s2r.load(sb_b0, k)
            sb0, sb1 = sb_all[0:2], sb_all[2:4]
            nk = k + 1
            prefetch = []
            for st in range_constexpr(N_LDS_STEPS_A):
                prefetch.append((a_g2s, a_nxt0, A0o + nk * KSTEP, st))
                prefetch.append((a_g2s, a_nxt1, A1o + nk * KSTEP, st))
            for st in range_constexpr(N_LDS_STEPS_B):
                prefetch.append((b_g2s, b_nxt0, B0o + nk * KSTEP, st))
                prefetch.append((b_g2s, b_nxt1, B1o + nk * KSTEP, st))
            quads = [(c00, a0, b0, sa0, sb0), (c01, a0, b1, sa0, sb1),
                     (c10, a1, b0, sa1, sb0), (c11, a1, b1, sa1, sb1)]
            il_mma(mfma, quads, prefetch, (False if k == 0 else asm_mfma), N_TILES_A, N_TILES_B, per, first=False, pinned=False, sched_mm=(1 if sched else 0), sched_dsrd_n=2 * (N_TILES_A + N_TILES_B))
            wait_barrier(0)
            a_cur0, a_nxt0 = a_nxt0, a_cur0
            a_cur1, a_nxt1 = a_nxt1, a_cur1
            b_cur0, b_nxt0 = b_nxt0, b_cur0
            b_cur1, b_nxt1 = b_nxt1, b_cur1

        kt = K_ITERS - 1
        a0 = a_s2r.load(a_cur0)
        a1 = a_s2r.load(a_cur1)
        b0 = b_s2r.load(b_cur0)
        b1 = b_s2r.load(b_cur1)
        sa0 = sa_s2r.load(sa_b0, kt)
        sa1 = sa_s2r.load(sa_b1, kt)
        sb_all = sb_s2r.load(sb_b0, kt)
        sb0, sb1 = sb_all[0:2], sb_all[2:4]
        quads = [(c00, a0, b0, sa0, sb0), (c01, a0, b1, sa0, sb1),
                 (c10, a1, b0, sa1, sb0), (c11, a1, b1, sa1, sb1)]
        il_mma(mfma, quads, [], (False if kt == 0 else asm_mfma), N_TILES_A, N_TILES_B, per, first=False, pinned=False, sched_mm=(1 if sched else 0), sched_dsrd_n=2 * (N_TILES_A + N_TILES_B))

        base_row = block_m * BLOCK_M + wm_off
        base_col = block_n * BLOCK_N + wn_off
        store_c.store(c00, base_row + 0, base_col + 0)
        store_c.store(c01, base_row + 0, base_col + LDS_BLOCK_N)
        store_c.store(c10, base_row + LDS_BLOCK_M, base_col + 0)
        store_c.store(c11, base_row + LDS_BLOCK_M, base_col + LDS_BLOCK_N)

    @flyc.kernel(known_block_size=[512, 1, 1])
    def kernel_gemm_rt2(
        A: fx.Tensor,
        B_T: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        c_m: fx.Int32,
        c_n: fx.Int32,
    ):
        # Double-buffered RUNTIME loop (unroll-2: ping-pong setA/setB with fixed
        # compile-time assignment per sub-iter, so no runtime ptr-select). Single
        # loop body -> iglp_opt(0) emitted once (no unroll explosion), and now the
        # prefetch G2S loads exist for the native iglp GEMM scheduler to interleave
        # among the MFMAs. Tests whether iglp compounds on a double-buffer base.
        F8_IR_t = fx.Float8E4M3FN.ir_type
        n_blocks = ceildiv(c_n, BLOCK_N)
        lds = fx.SharedAllocator().allocate(SharedStorageFp4Pipe).peek()
        # setA = cur, setB = next
        Aa0, Aa1, Ab0, Ab1 = lds.A_lds_cur_0, lds.A_lds_cur_1, lds.B_lds_cur_0, lds.B_lds_cur_1
        Ba0, Ba1, Bb0, Bb1 = lds.A_lds_next_0, lds.A_lds_next_1, lds.B_lds_next_0, lds.B_lds_next_1

        lane_id = fx.thread_idx.x % 64
        wave_id = fx.thread_idx.x // 64
        wave_m = wave_id // 4
        wave_n = wave_id % 4
        block_m, block_n = grouped_xcd_pid(fx.block_idx.x, c_m, c_n, BLOCK_M, BLOCK_N,
                                           group_m=group_m, num_xcds=num_xcds, group_n=group_n)

        gA = make_fp8_buffer_tensor(A, F8_IR_t)
        gB = make_fp8_buffer_tensor(B_T, F8_IR_t)
        a_div = fx.logical_divide(gA, fx.make_layout(1, 1))
        b_div = fx.logical_divide(gB, fx.make_layout(1, 1))
        gl_off_a = fp4_g2s_offsets(lane_id, wave_id, K, N_LDS_STEPS_A, BPR)
        gl_off_b = fp4_g2s_offsets(lane_id, wave_id, K, N_LDS_STEPS_B, BPR)
        mfma = MfmaScaleFp4(N_TILES_A, N_TILES_B)
        a_g2s = G2SLoader(a_div, gl_off_a, N_LDS_STEPS_A, F8_IR_t, wave_id)
        b_g2s = G2SLoader(b_div, gl_off_b, N_LDS_STEPS_B, F8_IR_t, wave_id)
        a_s2r = S2RLoaderFp4(wave_m, N_TILES_A, N_SUB, BPR, LDS_ROW_STRIDE)
        b_s2r = S2RLoaderFp4(wave_n, N_TILES_B, N_SUB, BPR, LDS_ROW_STRIDE)
        # A-scale num_records must cover the rows the kernel addresses =
        # ceil(c_m/BLOCK_M)*BLOCK_M (the padded extent). ScaleS2R floors
        # dim//group_span, so pass a group_span-ceil'd dim. For aligned M
        # (16*SA_TILES | c_m, true for all BM256 production shapes) this == c_m
        # (byte-identical); only BM192 (16*3=48 ∤ 4096) pads, covering the edge
        # M-tile's scale group (else valid rows 4080-4095 clamp to scale 0 -> SNR 24).
        _sa_q = 16 * SA_TILES
        sa_s2r = ScaleS2R(A_scale, ((c_m + _sa_q - 1) // _sa_q) * _sa_q, K, SA_TILES)
        sb_s2r = ScaleBComb(B_scale, c_n, K)
        store_c = (StoreCAtomic if split_k > 1 else StoreCPlain)(C, c_m, c_n, mfma.idx, N_TILES_A, N_TILES_B)
        wm_off = wave_m * (N_TILES_A * 16)
        wn_off = wave_n * (N_TILES_B * 16)
        sa_b0 = fx.Int32(block_m * BLOCK_M + wm_off)
        sa_b1 = sa_b0 + fx.Int32(LDS_BLOCK_M)
        sb_b0 = fx.Int32(block_n * BLOCK_N + wn_off)
        A0o = block_m * BLOCK_M * K2
        A1o = (block_m * BLOCK_M + LDS_BLOCK_M) * K2
        B0o = block_n * BLOCK_N * K2
        B1o = (block_n * BLOCK_N + LDS_BLOCK_N) * K2
        DSRD = 2 * (N_TILES_A + N_TILES_B)

        def g2s_set(sa0, sa1, sb0_, sb1_, ko):
            a_g2s.load(sa0, A0o + ko)
            a_g2s.load(sa1, A1o + ko)
            b_g2s.load(sb0_, B0o + ko)
            b_g2s.load(sb1_, B1o + ko)

        def body(ca0, ca1, cb0, cb1, na0, na1, nb0, nb1, ki, c00, c01, c10, c11):
            # compute (k=ki) from cur(set), prefetch k+2 into cur (after S2R), iglp, mma
            a0 = a_s2r.load(ca0); a1 = a_s2r.load(ca1)
            b0 = b_s2r.load(cb0); b1 = b_s2r.load(cb1)
            sa0v = sa_s2r.load(sa_b0, ki); sa1v = sa_s2r.load(sa_b1, ki)
            sb_all = sb_s2r.load(sb_b0, ki)
            sb0v, sb1v = sb_all[0:2], sb_all[2:4]
            wait_lgkm()  # S2R reads of cur drained before G2S overwrites it (WAR)
            g2s_set(ca0, ca1, cb0, cb1, (ki + 2) * KSTEP)  # prefetch into just-consumed cur
            if const_expr(iglp):
                rocdl.iglp_opt(0)
            quads = [(c00, a0, b0, sa0v, sb0v), (c01, a0, b1, sa0v, sb1v),
                     (c10, a1, b0, sa1v, sb0v), (c11, a1, b1, sa1v, sb1v)]
            il_mma(mfma, quads, [], False, N_TILES_A, N_TILES_B,
                   sched_mm=(1 if sched else 0), sched_dsrd_n=DSRD)
            wait_barrier(0)  # drain prefetch G2S (vmcnt) + WG-sync before buffer reused

        c00 = [mfma.zero_value] * N_ACCUMS
        c01 = [mfma.zero_value] * N_ACCUMS
        c10 = [mfma.zero_value] * N_ACCUMS
        c11 = [mfma.zero_value] * N_ACCUMS
        # prologue: setA<-k0, setB<-k1
        g2s_set(Aa0, Aa1, Ab0, Ab1, 0)
        g2s_set(Ba0, Ba1, Bb0, Bb1, KSTEP)
        wait_barrier(0)

        init_args = (c00 + c01 + c10 + c11)
        loop_results = init_args
        for kk, args in range(0, K_ITERS, 2, init=init_args):
            ki = arith.index_cast(T.i32, kk)
            c00 = list(args[0 * N_ACCUMS:1 * N_ACCUMS])
            c01 = list(args[1 * N_ACCUMS:2 * N_ACCUMS])
            c10 = list(args[2 * N_ACCUMS:3 * N_ACCUMS])
            c11 = list(args[3 * N_ACCUMS:4 * N_ACCUMS])
            body(Aa0, Aa1, Ab0, Ab1, Ba0, Ba1, Bb0, Bb1, ki, c00, c01, c10, c11)
            body(Ba0, Ba1, Bb0, Bb1, Aa0, Aa1, Ab0, Ab1, ki + 1, c00, c01, c10, c11)
            loop_results = yield (c00 + c01 + c10 + c11)

        c00 = loop_results[0 * N_ACCUMS:1 * N_ACCUMS]
        c01 = loop_results[1 * N_ACCUMS:2 * N_ACCUMS]
        c10 = loop_results[2 * N_ACCUMS:3 * N_ACCUMS]
        c11 = loop_results[3 * N_ACCUMS:4 * N_ACCUMS]
        base_row = block_m * BLOCK_M + wm_off
        base_col = block_n * BLOCK_N + wn_off
        store_c.store(c00, base_row + 0, base_col + 0)
        store_c.store(c01, base_row + 0, base_col + LDS_BLOCK_N)
        store_c.store(c10, base_row + LDS_BLOCK_M, base_col + 0)
        store_c.store(c11, base_row + LDS_BLOCK_M, base_col + LDS_BLOCK_N)

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
        n_blocks = ceildiv(c_n, BLOCK_N)
        wave_id = fx.thread_idx.x // 64
        wave_m = wave_id // 4
        wave_n = wave_id % 4
        block_m, block_n = divmod(fx.block_idx.x, n_blocks)

        mfma = MfmaScaleFp4(N_TILES_A, N_TILES_B, asm=asm_mfma, asm_se=asm_se)
        a_ld = Fp4FragLoader(A, c_m, K, N_TILES_A)
        b_ld = Fp4FragLoader(B_T, c_n, K, N_TILES_B)
        # A-scale num_records must cover the rows the kernel addresses =
        # ceil(c_m/BLOCK_M)*BLOCK_M (the padded extent). ScaleS2R floors
        # dim//group_span, so pass a group_span-ceil'd dim. For aligned M
        # (16*SA_TILES | c_m, true for all BM256 production shapes) this == c_m
        # (byte-identical); only BM192 (16*3=48 ∤ 4096) pads, covering the edge
        # M-tile's scale group (else valid rows 4080-4095 clamp to scale 0 -> SNR 24).
        _sa_q = 16 * SA_TILES
        sa_s2r = ScaleS2R(A_scale, ((c_m + _sa_q - 1) // _sa_q) * _sa_q, K, SA_TILES)
        sb_s2r = ScaleBComb(B_scale, c_n, K) if B_COMB else ScaleBRegion(B_scale, c_n, K, N_TILES_B)
        store_c = (StoreCAtomic if split_k > 1 else StoreCPlain)(C, c_m, c_n, mfma.idx, N_TILES_A, N_TILES_B)

        wave_m_offset = wave_m * (N_TILES_A * 16)
        wave_n_offset = wave_n * (N_TILES_B * 16)
        a_base0 = fx.Int32(block_m * BLOCK_M + wave_m_offset)
        a_base1 = a_base0 + fx.Int32(LDS_BLOCK_M)
        b_base0 = fx.Int32(block_n * BLOCK_N + wave_n_offset)
        b_base1 = b_base0 + fx.Int32(LDS_BLOCK_N)

        c00 = [mfma.zero_value] * N_ACCUMS
        c01 = [mfma.zero_value] * N_ACCUMS
        c10 = [mfma.zero_value] * N_ACCUMS
        c11 = [mfma.zero_value] * N_ACCUMS

        for k in range_constexpr(K_ITERS):
            a0 = a_ld.load(a_base0, k)
            a1 = a_ld.load(a_base1, k)
            b0 = b_ld.load(b_base0, k)
            b1 = b_ld.load(b_base1, k)
            sa0 = sa_s2r.load(a_base0, k)
            sa1 = sa_s2r.load(a_base1, k)
            if const_expr(B_COMB):
                sb_all = sb_s2r.load(b_base0, k)
                sb0, sb1 = sb_all[0:2], sb_all[2:4]
            else:
                sb0 = sb_s2r.load(b_base0, k)
                sb1 = sb_s2r.load(b_base1, k)

            c00 = mfma.call(a0, b0, c00, sa0, sb0)
            c01 = mfma.call(a0, b1, c01, sa0, sb1)
            c10 = mfma.call(a1, b0, c10, sa1, sb0)
            c11 = mfma.call(a1, b1, c11, sa1, sb1)

        base_row = block_m * BLOCK_M + wave_m_offset
        base_col = block_n * BLOCK_N + wave_n_offset
        store_c.store(c00, base_row + 0, base_col + 0)
        store_c.store(c01, base_row + 0, base_col + LDS_BLOCK_N)
        store_c.store(c10, base_row + LDS_BLOCK_M, base_col + 0)
        store_c.store(c11, base_row + LDS_BLOCK_M, base_col + LDS_BLOCK_N)

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
        grid_x = ceildiv(c_m, BLOCK_M) * ceildiv(c_n, BLOCK_N) * split_k
        kern = {"staged": kernel_gemm_staged, "pipe": kernel_gemm_pipe, "pipeh": kernel_gemm_pipeh,
                "pipe3": kernel_gemm_pipe3,
                "pipeb": kernel_gemm_pipeb,
                "il": kernel_gemm_il, "rt": kernel_gemm_rt, "rt2": kernel_gemm_rt2}.get(mode, kernel_gemm)
        kern(
            A, B_T, C, A_scale, B_scale, c_m, c_n,
            value_attrs={"rocdl.waves_per_eu": 2, "rocdl.flat_work_group_size": "512,512"},
        ).launch(grid=(grid_x, 1, 1), block=(512, 1, 1), stream=stream)

    return launch_gemm
