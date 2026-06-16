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
from flydsl.expr import arith, buffer_ops, const_expr, primitive, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from kernels.fp8_gemm_utils import (
    G2SLoader,
    ceildiv,
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


def _raw(v):
    if not isinstance(v, ir.Value) and hasattr(v, "ir_value"):
        return v.ir_value()
    return v


def _packed_eligible(mode, BLOCK_N):
    """Whether the packed-scale (opsel) path applies: production pipe, BN256
    combined-B. Single source of truth shared by the kernel
    (compile_mxfp4_gemm_8w) and the host scale prep."""
    return mode in ("pipe", "wholeloop") and BLOCK_N >= 256


def preshuffle_mxfp4_scales(a_e8m0, b_e8m0, K, BLOCK_M=256, BLOCK_N=256,
                            mode="pipe", combine_a=False):
    """Host scale prep matching the kernel's chosen layout: PACKED (opsel) for the
    eligible production pipe path, else broadcast-dwordx4. combine_a packs both
    M-regions' A scale into one coalesced dwordx2 (default). Returns
    (a_scale_i32, b_scale_i32) -- keeps host & kernel in lockstep."""
    n_tiles_a = BLOCK_M // 64
    if _packed_eligible(mode, BLOCK_N):
        a_pk = (preshuffle_scale_packed_a2(a_e8m0, K, n_tiles_a)
                if (combine_a and BLOCK_M == 256 and mode != "wholeloop")
                else preshuffle_scale_packed(a_e8m0, K, n_tiles_a))
        return a_pk, preshuffle_scale_b_comb_packed(b_e8m0, K)
    a_bc = preshuffle_scale(a_e8m0, K, n_tiles_a)
    if BLOCK_N >= 256:  # combined-B (B_COMB)
        return a_bc, preshuffle_scale_b_comb(b_e8m0, K)
    return a_bc, preshuffle_scale(b_e8m0, K, BLOCK_N // 128)  # BN128: per-region B (ScaleBRegion)


def preshuffle_scale_lane_contig(e8m0_u8, K, n_tiles, n_sub, mode):
    """VGPR-direct (gluon-style 128-bit contiguous scale read): re-layout the PACKED
    scale so a lane's 2*n_sub packed dwords (2 region groups x n_sub) are CONTIGUOUS in
    gmem -> ONE buffer_load_dwordx{2*n_sub} into VGPR (no LDS/ds_write).

    The two region groups a wave covers differ by mode:
      mode='A': consecutive groups (g0, g0+1).      wi = base//128, g0 = 2*wi.
      mode='B': stride-2 groups (g0, g0+2) with block interleave (BL/BR are +128 cols =
                +2 groups). wi = block*2+wave_n, g0 = (wi//2)*4 + (wi%2).
    Output int32 [n_wi, K//128//n_sub, 64, 2*n_sub]; last dim = [r*n_sub + s] which equals
    the kernel's sa_t/sb_t index (g*n_sub + s). soffset = wi*K128*512 bytes (kk stride 1024).
    """
    import torch
    p = preshuffle_scale_packed(e8m0_u8, K, n_tiles)   # [G, K128, 64] i32
    G, K128, _ = p.shape
    assert K128 % n_sub == 0
    n_wi = G // 2; KK = K128 // n_sub; nd = 2 * n_sub
    out = torch.empty((n_wi, KK, 64, nd), dtype=torch.int32, device=p.device)
    for wi in range(n_wi):
        if mode == 'A':
            groups = [2 * wi, 2 * wi + 1]
        else:  # 'B'
            g0 = (wi // 2) * 4 + (wi % 2)
            groups = [g0, g0 + 2]
        for r, g in enumerate(groups):
            pr = p[g].reshape(KK, n_sub, 64)        # [kk, s, lane]
            for s in range(n_sub):
                out[wi, :, :, r * n_sub + s] = pr[:, s, :]
    return out.contiguous()


def preshuffle_scale_packed(e8m0_u8, K, n_tiles):
    """PACKED E8M0 pre-shuffle (official CK-style): pack the wave's ``n_tiles``
    sub-tile scales into the 4 BYTES of ONE i32 (byte t = tile t's E8M0), instead
    of broadcasting each tile to its own dword. The MFMA then selects tile t's
    scale via opsel_a=t -- so the kernel loads ONE dword per (region,k) instead of
    a dwordx4 (4x less scale VMEM traffic / VMEM-unit occupancy, the bulk lever).

    Input : uint8 [DIM, K//32] (DIM multiple of 16*n_tiles, n_tiles<=4).
    Output: int32 [DIM//(16*n_tiles), K//128, 64] where byte t of SP[grp,k,lane]
        == scale[grp*16*n_tiles + t*16 + lane%16, 4k + lane//16].
    """
    import torch

    DIM, Kb = e8m0_u8.shape
    assert Kb == K // 32 and K % 128 == 0 and n_tiles <= 4
    assert DIM % (16 * n_tiles) == 0, f"DIM={DIM} must be multiple of {16 * n_tiles}"
    K128 = K // 128
    G = DIM // (16 * n_tiles)
    s = e8m0_u8.reshape(DIM, K128, 4)                       # [DIM, k, g]
    s = s.reshape(G, n_tiles, 16, K128, 4)                  # [grp, t, r, k, g]
    s = s.permute(0, 3, 4, 2, 1).contiguous()              # [grp, k, g, r, t]
    s = s.reshape(G, K128, 64, n_tiles).to(torch.int32)    # lane == g*16 + r
    packed = s[..., 0].clone()
    for t in range(1, n_tiles):
        packed = packed | (s[..., t] << (8 * t))
    return packed.contiguous()                              # [G, K128, 64] i32


def preshuffle_scale_b_comb_packed(e8m0_u8, K):
    """Combined-B PACKED pre-shuffle: pack a wave's 4 B sub-tiles (b0:0,16; b1:128,
    144) into the 4 bytes of ONE i32 (byte i = B-tile i). MFMA selects via opsel_b
    (b0 -> opsel 0,1; b1 -> opsel 2,3). One dword per K-iter for all B scales.

    Output: int32 [N//64, K//128, 64], byte i of SP[grp,k,lane]
        == scale[block_n*256 + wave_n*32 + OFF[i] + lane%16, 4k + lane//16].
    """
    import torch

    N, Kb = e8m0_u8.shape
    assert Kb == K // 32 and K % 128 == 0 and N % 256 == 0
    K128 = K // 128
    OFF = [0, 16, 128, 144]
    s = e8m0_u8.reshape(N // 256, 256, K128, 4)
    wn = torch.arange(4).view(4, 1, 1)
    si = torch.arange(4).view(1, 4, 1)
    r = torch.arange(16).view(1, 1, 16)
    off = torch.tensor(OFF).view(1, 4, 1)
    colidx = (wn * 32 + off + r).reshape(-1)
    g = s[:, colidx, :, :].reshape(N // 256, 4, 4, 16, K128, 4)  # [nblk, wn, si, r, k, g]
    g = g.permute(0, 1, 4, 5, 3, 2).contiguous()                # [nblk, wn, k, g, r, si]
    g = g.reshape(N // 64, K128, 64, 4).to(torch.int32)         # grp=nblk*4+wn, lane=g*16+r
    packed = g[..., 0] | (g[..., 1] << 8) | (g[..., 2] << 16) | (g[..., 3] << 24)
    return packed.contiguous()                                  # [N//64, K128, 64] i32


def preshuffle_scale_packed_a2(e8m0_u8, K, n_tiles):
    """Combine-A packed layout: interleave the TWO M-regions' packed A scale into a
    COALESCED dwordx2 so the kernel issues ONE load (instead of two) for both regions
    per K-iter. region0 = group g0, region1 = g0+2 (region1 is +LDS_BLOCK_M=+128 rows
    = +2 groups). Output int32 [n_wi, K//128, 64, 2] (wi = block_m*2 + wave_m), last
    dim {region0, region1}. Coalesced (lane-consecutive) -- unlike the K-batch
    transpose; cuts the 3 scale loads/iter to 2 without breaking coalescing."""
    import torch

    p = preshuffle_scale_packed(e8m0_u8, K, n_tiles)   # [G, K128, 64]
    G, K128, _ = p.shape
    assert G % 2 == 0, f"combine-A needs even group count, got G={G}"
    n_wi = G // 2
    out = torch.empty((n_wi, K128, 64, 2), dtype=torch.int32, device=p.device)
    # Wave's two A regions are CONSECUTIVE 64-row groups (g0, g0+1); g0=2*wi is even
    # (g0 = sa_base//64 = block_m*4 + wave_m*2). So wi = g0//2 = sa_base//128.
    for wi in range(n_wi):
        out[wi, :, :, 0] = p[2 * wi]
        out[wi, :, :, 1] = p[2 * wi + 1]
    return out.contiguous()


class ScaleS2RPackedA2:
    """Combine-A loader: ONE coalesced dwordx2 returns both M-regions' packed scale
    [region0_dword, region1_dword] for a (wi, k, lane). Pairs with
    preshuffle_scale_packed_a2. base is the region0 base; wi derived from it."""

    def __init__(self, sp_tensor, dim, K, n_tiles):
        self.K128 = K // 128
        self.group_span = 16 * n_tiles
        self.lane = fx.thread_idx.x % 64
        n_wi = (dim // self.group_span) // 2
        nbytes = n_wi * self.K128 * 64 * 2 * 4
        self.rsrc = buffer_ops.create_buffer_resource(sp_tensor, max_size=False, num_records_bytes=nbytes)

    def load(self, base0, k):
        wi = base0 // 128                                # g0//2, g0 = base0//64 (even)
        idx = ((wi * self.K128 + k) * 64 + self.lane) * 2
        v = Vec(buffer_ops.buffer_load(self.rsrc, idx, vec_width=2, dtype=T.i32))
        return [_raw(v[0]), _raw(v[1])]


class ScaleS2RPacked:
    """Packed-scale loader: ONE dword per (region, k) holding n_tiles E8M0 scales
    (byte t = tile t). Pairs with ``preshuffle_scale_packed``. The tile index is
    consumed as the MFMA opsel, so .load returns a single raw i32 (the packed dword)."""

    def __init__(self, sp_tensor, dim, K, n_tiles):
        self.K128 = K // 128
        self.n_tiles = n_tiles
        self.group_span = 16 * n_tiles
        self.lane = fx.thread_idx.x % 64
        nbytes = (dim // self.group_span) * self.K128 * 64 * 4  # int32 records, 1/lane
        self.rsrc = buffer_ops.create_buffer_resource(sp_tensor, max_size=False, num_records_bytes=nbytes)

    def load(self, base, k):
        grp = base // self.group_span
        idx = (grp * self.K128 + k) * 64 + self.lane
        return _raw(buffer_ops.buffer_load(self.rsrc, idx, vec_width=1, dtype=T.i32))


class ScaleBCombPacked:
    """Combined-B packed loader: ONE dword/(k) holding all 4 B sub-tile scales
    (b0:bytes 0,1; b1:bytes 2,3). Pairs with ``preshuffle_scale_b_comb_packed``."""

    def __init__(self, sp_tensor, dim, K):
        self.K128 = K // 128
        self.lane = fx.thread_idx.x % 64
        nbytes = (dim // 64) * self.K128 * 64 * 4
        self.rsrc = buffer_ops.create_buffer_resource(sp_tensor, max_size=False, num_records_bytes=nbytes)

    def load(self, base, k):
        # grp = block_n*4 + wave_n (matches ScaleBComb / preshuffle_scale_b_comb_packed).
        grp = (base // 256) * 4 + (base % 256) // 32
        idx = (grp * self.K128 + k) * 64 + self.lane
        return _raw(buffer_ops.buffer_load(self.rsrc, idx, vec_width=1, dtype=T.i32))


class MfmaScaleFp4:
    """16x16x128 f8f6f4 MFMA in fp4 mode (cbsz=4/blgp=4) with per-block E8M0 scales.
    packed=True: ONE packed-i32 scale operand per region, opsel selects the per-XDL
    byte (4x less scale VMEM than broadcast-dwordx4). Intrinsic MFMA; a/b are i32x8
    (low 16B real, upper zero) from the S2R frag loader."""

    def __init__(self, n_tiles_a, n_tiles_b, packed=False):
        self.res_ty = Vec.make_type(4, fx.Float32)
        self.zero_value = Vec.filled(4, 0.0, fx.Float32)
        self.n_tiles_a = n_tiles_a
        self.n_tiles_b = n_tiles_b
        self.packed = packed

    def idx(self, i, j):
        return i * self.n_tiles_b + j

    def _do(self, a, b, c, sa, sb):
        return rocdl.mfma_scale_f32_16x16x128_f8f6f4(self.res_ty, [a, b, c, 4, 4, 0, sa, 0, sb])

    def call(self, a, b, c, sa, sb):
        for i in range_constexpr(self.n_tiles_a):
            for j in range_constexpr(self.n_tiles_b):
                c[self.idx(i, j)] = self._do(a[i], b[j], c[self.idx(i, j)], sa[i], sb[j])
        return c

    def _do_packed(self, a, b, c, sa_p, opsel_a, sb_p, opsel_b):
        """PACKED-scale MFMA: one i32 scale operand holds n_tiles E8M0 (byte t = tile
        t); opsel selects the byte for this XDL."""
        return rocdl.mfma_scale_f32_16x16x128_f8f6f4(
            self.res_ty, [a, b, c, 4, 4, opsel_a, sa_p, opsel_b, sb_p])

    def call_subs(self, a, b, c, sa, sb, n_sub):
        """Accumulate over n_sub 128-K sub-blocks. a[i][s]/b[j][s] nested frags.
        packed: sa[s] = ONE packed i32 (byte t = A-tile t); sb[s] = tuple
        (packed_i32, opsel_b_base). non-packed: sa[s]/sb[s] = per-tile i32 lists."""
        if self.packed:
            for s in range_constexpr(n_sub):
                sb_p, ob = sb[s]
                for i in range_constexpr(self.n_tiles_a):
                    for j in range_constexpr(self.n_tiles_b):
                        c[self.idx(i, j)] = self._do_packed(
                            a[i][s], b[j][s], c[self.idx(i, j)], sa[s], i, sb_p, ob + j)
            return c
        for s in range_constexpr(n_sub):
            a_s = [a[i][s] for i in range_constexpr(self.n_tiles_a)]
            b_s = [b[j][s] for j in range_constexpr(self.n_tiles_b)]
            c = self.call(a_s, b_s, c, sa[s], sb[s])
        return c

    def call_subs_asm(self, a, b, c, sa, sb, n_sub, _cache={}):
        """asm-cluster version of call_subs (packed scale): emit all n_sub*nta*ntb
        v_mfma_scale in ONE opaque inline-asm block with =&v tied accumulators. Opaque
        to the backend -> lets a manual s_waitcnt vmcnt(N) (issued around it) stagger
        B-load overlap (intrinsic MFMA's auto-vmcnt(0) defeats it). a[i][s]/b[j][s] i32x8,
        sa[s] packed-i32, sb[s]=(packed_i32, opsel_b_base). Returns updated c (list)."""
        assert self.packed
        nta, ntb = self.n_tiles_a, self.n_tiles_b
        Nacc = nta * ntb
        ob = sb[0][1]  # opsel_b_base (compile-time const per region: 0 or 2)
        na, nb = nta * n_sub, ntb * n_sub
        _agpr = __import__("os").environ.get("FP4_AGPR", "0") == "1"
        key = (nta, ntb, n_sub, ob, _agpr)
        if key not in _cache:
            base_a, base_b = Nacc, Nacc + na
            base_sa, base_sb = Nacc + na + nb, Nacc + na + nb + n_sub
            L = []
            for s in range(n_sub):
                for i in range(nta):
                    for j in range(ntb):
                        q = i * ntb + j
                        oa, obj = i, ob + j
                        osel = (f"op_sel:[{oa & 1},{obj & 1},0] "
                                f"op_sel_hi:[{(oa >> 1) & 1},{(obj >> 1) & 1},0]")
                        L.append(f"v_mfma_scale_f32_16x16x128_f8f6f4 ${q}, ${base_a + i * n_sub + s}, "
                                 f"${base_b + j * n_sub + s}, ${q}, ${base_sa + s}, ${base_sb + s} "
                                 f"{osel} cbsz:4 blgp:4")
            _ac = "=a" if _agpr else "=v"
            cons = ",".join([_ac] * Nacc + ["v"] * (na + nb + 2 * n_sub) + [str(q) for q in range(Nacc)])
            st = "!llvm.struct<(" + ", ".join(["vector<4xf32>"] * Nacc) + ")>"
            _cache[key] = ("\n".join(L), cons, st)
        asm, cons, st = _cache[key]
        ins = []
        for i in range_constexpr(nta):
            for s in range_constexpr(n_sub):
                ins.append(_raw(a[i][s]))
        for j in range_constexpr(ntb):
            for s in range_constexpr(n_sub):
                ins.append(_raw(b[j][s]))
        for s in range_constexpr(n_sub):
            ins.append(_raw(sa[s]))
        for s in range_constexpr(n_sub):
            ins.append(_raw(sb[s][0]))
        for q in range_constexpr(Nacc):
            ins.append(_raw(c[q]))
        r = _llvm.inline_asm(ir.Type.parse(st), ins, asm, cons, has_side_effects=True)
        return [Vec(_llvm.extractvalue(ir.Type.parse("vector<4xf32>"), r, [q])) for q in range_constexpr(Nacc)]

    def call_packed_asm(self, a, b, c, sa, sb, n_sub, _cache={}):
        """4-wave packed-scale asm MFMA cluster (one N-slice = nta*ntb accs). nta(8)
        A-tiles need 2 scale dwords: sa[s][i//4], opsel i%4. sb[s] one dword, opsel j.
        accs =a AGPR tied. Opaque -> forces MFMA order so a hand-interleave (G2S/ds_read
        spread into the cluster) survives the backend (vs compiler re-scheduling). Mirrors
        the 4-wave _mfma_packed math; a[i][s]/b[j][s] i32x4 (pad=False), sa[s][g]/sb[s] i32."""
        assert self.packed
        nta, ntb = self.n_tiles_a, self.n_tiles_b
        Nacc = nta * ntb
        na, nb = nta * n_sub, ntb * n_sub
        nsa, nsb = 2 * n_sub, n_sub
        _agpr = __import__("os").environ.get("FP4_AGPR", "1") == "1"
        key = (nta, ntb, n_sub, _agpr)
        if key not in _cache:
            base_a = Nacc
            base_b = base_a + na
            base_sa = base_b + nb
            base_sb = base_sa + nsa
            L = []
            for s in range(n_sub):
                for i in range(nta):
                    for j in range(ntb):
                        q = i * ntb + j
                        oa, ob = i % 4, j
                        sa_op = base_sa + s * 2 + i // 4
                        sb_op = base_sb + s
                        osel = (f"op_sel:[{oa & 1},{ob & 1},0] "
                                f"op_sel_hi:[{(oa >> 1) & 1},{(ob >> 1) & 1},0]")
                        L.append(f"v_mfma_scale_f32_16x16x128_f8f6f4 ${q}, ${base_a + i * n_sub + s}, "
                                 f"${base_b + j * n_sub + s}, ${q}, ${sa_op}, ${sb_op} {osel} cbsz:4 blgp:4")
            _ac = "=a" if _agpr else "=v"
            cons = ",".join([_ac] * Nacc + ["v"] * (na + nb + nsa + nsb) + [str(q) for q in range(Nacc)])
            st = "!llvm.struct<(" + ", ".join(["vector<4xf32>"] * Nacc) + ")>"
            _cache[key] = ("\n".join(L), cons, st)
        asm, cons, st = _cache[key]
        ins = []
        for i in range_constexpr(nta):
            for s in range_constexpr(n_sub):
                ins.append(_raw(a[i][s]))
        for j in range_constexpr(ntb):
            for s in range_constexpr(n_sub):
                ins.append(_raw(b[j][s]))
        for s in range_constexpr(n_sub):
            for g in range_constexpr(2):
                ins.append(_raw(sa[s][g]))
        for s in range_constexpr(n_sub):
            ins.append(_raw(sb[s]))
        for q in range_constexpr(Nacc):
            ins.append(_raw(c[q]))
        r = _llvm.inline_asm(ir.Type.parse(st), ins, asm, cons, has_side_effects=True)
        return [Vec(_llvm.extractvalue(ir.Type.parse("vector<4xf32>"), r, [q])) for q in range_constexpr(Nacc)]

    def call_packed_asm2(self, a, bl, br, cL, cR, sa, sbl, sbr, n_sub, _cache={}):
        """Combined 2-N-slice 128-MFMA asm cluster (accL+accR in ONE block, 64 accs =a
        AGPR). Removes the accL->accR boundary gap a 2-cluster split leaves, and is the
        carrier for interleaving ds_read/G2S between MFMAs (gluon-density). a[i][s] shared;
        bl/br[j][s] per-slice; sa[s][g] shared; sbl/sbr[s] per-slice."""
        assert self.packed
        nta, ntb = self.n_tiles_a, self.n_tiles_b
        nq = nta * ntb                       # 32 accs/slice
        NT = 2 * nq                          # 64
        na = nta * n_sub
        nb = ntb * n_sub
        _agpr = __import__("os").environ.get("FP4_AGPR", "1") == "1"
        key = (nta, ntb, n_sub, _agpr)
        if key not in _cache:
            base_a = NT
            base_bl = base_a + na
            base_br = base_bl + nb
            base_sa = base_br + nb
            base_sbl = base_sa + 2 * n_sub
            base_sbr = base_sbl + n_sub
            L = []
            for (sl, b_base, sb_base) in ((0, base_bl, base_sbl), (1, base_br, base_sbr)):
                for s in range(n_sub):
                    for i in range(nta):
                        for j in range(ntb):
                            q = sl * nq + i * ntb + j
                            oa, ob = i % 4, j
                            sa_op = base_sa + s * 2 + i // 4
                            osel = (f"op_sel:[{oa & 1},{ob & 1},0] "
                                    f"op_sel_hi:[{(oa >> 1) & 1},{(ob >> 1) & 1},0]")
                            L.append(f"v_mfma_scale_f32_16x16x128_f8f6f4 ${q}, ${base_a + i * n_sub + s}, "
                                     f"${b_base + j * n_sub + s}, ${q}, ${sa_op}, ${sb_base + s} {osel} cbsz:4 blgp:4")
            _ac = "=a" if _agpr else "=v"
            cons = ",".join([_ac] * NT + ["v"] * (na + 2 * nb + 4 * n_sub) + [str(q) for q in range(NT)])
            st = "!llvm.struct<(" + ", ".join(["vector<4xf32>"] * NT) + ")>"
            _cache[key] = ("\n".join(L), cons, st)
        asm, cons, st = _cache[key]
        ins = []
        for i in range_constexpr(nta):
            for s in range_constexpr(n_sub):
                ins.append(_raw(a[i][s]))
        for fr in (bl, br):
            for j in range_constexpr(ntb):
                for s in range_constexpr(n_sub):
                    ins.append(_raw(fr[j][s]))
        for s in range_constexpr(n_sub):
            for g in range_constexpr(2):
                ins.append(_raw(sa[s][g]))
        for sc in (sbl, sbr):
            for s in range_constexpr(n_sub):
                ins.append(_raw(sc[s]))
        for q in range_constexpr(nq):
            ins.append(_raw(cL[q]))
        for q in range_constexpr(nq):
            ins.append(_raw(cR[q]))
        r = _llvm.inline_asm(ir.Type.parse(st), ins, asm, cons, has_side_effects=True)
        out = [Vec(_llvm.extractvalue(ir.Type.parse("vector<4xf32>"), r, [q])) for q in range_constexpr(NT)]
        return out[:nq], out[nq:]

    def call_packed_asm2_g2s(self, a, bl, br, cL, cR, sa, sbl, sbr, n_sub, g2s, _cache={}):
        """HYBRID hot loop: combined 128-MFMA cluster (call_packed_asm2) with the next-iter
        G2S (buffer_load_lds) hand-interleaved into the MFMA stream -> mem overlaps MFMA
        like gluon's LLIR-sched, but I place every instruction (no compiler in the loop).
        g2s = list of (lds_ptr_i32, voff_i32, rsrc_ptr8, soff_i32) prefetch steps; spread
        across the SECOND half of the 128 MFMA (after current reads consumed -> WAR-safe).
        accs =a AGPR. Returns (accL, accR)."""
        assert self.packed
        nta, ntb = self.n_tiles_a, self.n_tiles_b
        nq = nta * ntb; NT = 2 * nq
        na, nb = nta * n_sub, ntb * n_sub
        ng = len(g2s)
        _agpr = __import__("os").environ.get("FP4_AGPR", "1") == "1"
        key = (nta, ntb, n_sub, ng, _agpr)
        if key not in _cache:
            base_a = NT; base_bl = base_a + na; base_br = base_bl + nb
            base_sa = base_br + nb; base_sbl = base_sa + 2 * n_sub; base_sbr = base_sbl + n_sub
            base_g = base_sbr + n_sub          # G2S operands start (4 per step: lds,voff,rsrc,soff)
            mlines = []
            for (sl, b_base, sb_base) in ((0, base_bl, base_sbl), (1, base_br, base_sbr)):
                for s in range(n_sub):
                    for i in range(nta):
                        for j in range(ntb):
                            q = sl * nq + i * ntb + j
                            oa, ob = i % 4, j
                            sa_op = base_sa + s * 2 + i // 4
                            osel = (f"op_sel:[{oa & 1},{ob & 1},0] op_sel_hi:[{(oa >> 1) & 1},{(ob >> 1) & 1},0]")
                            mlines.append(f"v_mfma_scale_f32_16x16x128_f8f6f4 ${q}, ${base_a + i*n_sub+s}, "
                                          f"${b_base + j*n_sub+s}, ${q}, ${sa_op}, ${sb_base+s} {osel} cbsz:4 blgp:4")
            # interleave: spread ng G2S steps across the 2nd half of the MFMA stream.
            glines = []
            for t in range(ng):
                o = base_g + t * 4   # 4 operands/step: lds(m0,s), voff(v), rsrc(s), soff(s). FlyDSL form: no sc0.
                glines.append(f"s_mov_b32 m0, ${o}\nbuffer_load_dwordx4 ${o+1}, ${o+2}, ${o+3} offen lds")
            # Spread G2S across the WHOLE MFMA stream (earlier issue -> more overlap; WAR-safe:
            # G2S writes next-iter buffers, not this iter's reads). FP4_G2SHALF=1 -> 2nd half only.
            out = []
            start = (len(mlines) // 2) if int(__import__("os").environ.get("FP4_G2SHALF", "0")) else 0
            gap = max((len(mlines) - start) // max(ng, 1), 1)
            gi = 0
            for idx, ml in enumerate(mlines):
                out.append(ml)
                if idx >= start and gi < ng and (idx - start) % gap == 0:
                    out.append(glines[gi]); gi += 1
            while gi < ng:
                out.append(glines[gi]); gi += 1
            _ac = "=a" if _agpr else "=v"
            cons = ",".join([_ac]*NT + ["v"]*(na+2*nb+4*n_sub) + ["s","v","s","s"]*ng + [str(q) for q in range(NT)])
            st = "!llvm.struct<(" + ", ".join(["vector<4xf32>"]*NT) + ")>"
            _cache[key] = ("\n".join(out), cons, st)
        asm, cons, st = _cache[key]
        ins = []
        for i in range_constexpr(nta):
            for s in range_constexpr(n_sub):
                ins.append(_raw(a[i][s]))
        for fr in (bl, br):
            for j in range_constexpr(ntb):
                for s in range_constexpr(n_sub):
                    ins.append(_raw(fr[j][s]))
        for s in range_constexpr(n_sub):
            for gg in range_constexpr(2):
                ins.append(_raw(sa[s][gg]))
        for sc in (sbl, sbr):
            for s in range_constexpr(n_sub):
                ins.append(_raw(sc[s]))
        for (lds_p, voff, rsrc, soff) in g2s:
            ins.append(_raw(lds_p)); ins.append(_raw(voff)); ins.append(_raw(rsrc)); ins.append(_raw(soff))
        for q in range_constexpr(nq):
            ins.append(_raw(cL[q]))
        for q in range_constexpr(nq):
            ins.append(_raw(cR[q]))
        r = _llvm.inline_asm(ir.Type.parse(st), ins, asm, cons, has_side_effects=True)
        o = [Vec(_llvm.extractvalue(ir.Type.parse("vector<4xf32>"), r, [q])) for q in range_constexpr(NT)]
        return o[:nq], o[nq:]

    def call_packed_asm2_g2s_dsr(self, a_base, bl_base, br_base, ts_a, ts_b,
                                 cL, cR, sa, sbl, sbr, n_sub, g2s, _cache={}):
        """HYBRID hot loop v2: like call_packed_asm2_g2s, but the operand ds_read is ALSO
        emitted INSIDE the asm (vs FlyDSL-issued outside -> a hard lgkmcnt(0) barrier that
        exposes the FULL read latency before any MFMA). Here I issue all ntmp ds_read_b128
        first (read_order = MFMA first-use), then run the 128-MFMA stream with STAGGERED
        s_waitcnt lgkmcnt(N): MFMA m only waits for its own frags (read early) -> the read
        latency overlaps the MFMA (MFMA >> ds_read latency). This is the 4915->5759 lever.
        a_base/bl_base/br_base = list of n_sub per-sub tile-0 LDS byte-addresses (i32, VGPR);
        tile i frag at base + i*ts_{a,b} (ts a 1024-multiple -> swizzle-safe). Same temps as
        the operand frags (128 VGPR) -> no extra register pressure. accs =a AGPR."""
        assert self.packed
        nta, ntb = self.n_tiles_a, self.n_tiles_b
        nq = nta * ntb; NT = 2 * nq
        na, nb = nta * n_sub, ntb * n_sub
        ntmp = na + 2 * nb                 # 32: per-(region,tile,sub) ds_read temps
        ng = len(g2s)
        _agpr = __import__("os").environ.get("FP4_AGPR", "1") == "1"
        key = (nta, ntb, n_sub, ng, ts_a, ts_b, _agpr)
        if key not in _cache:
            t_a = NT                        # temp output bases (after NT acc outputs)
            t_bl = t_a + na; t_br = t_bl + nb
            # input operand bases (after NT+ntmp outputs):
            in0 = NT + ntmp
            i_ab = in0; i_blb = i_ab + n_sub; i_brb = i_blb + n_sub
            i_sa = i_brb + n_sub; i_sbl = i_sa + 2 * n_sub; i_sbr = i_sbl + n_sub
            i_g = i_sbr + n_sub

            def temp_a(i, s): return t_a + i * n_sub + s
            def temp_b(base, j, s): return base + j * n_sub + s

            # MFMA stream metadata (slice L then R); collect frag first-use -> read_order.
            mfmas = []   # (q, a_temp, b_temp, sa_op, sb_in, osel)
            for (sl, b_tbase, sb_in_base) in ((0, t_bl, i_sbl), (1, t_br, i_sbr)):
                for s in range(n_sub):
                    for i in range(nta):
                        for j in range(ntb):
                            q = sl * nq + i * ntb + j
                            oa, ob = i % 4, j
                            sa_op = i_sa + s * 2 + i // 4
                            osel = (f"op_sel:[{oa & 1},{ob & 1},0] op_sel_hi:[{(oa >> 1) & 1},{(ob >> 1) & 1},0]")
                            mfmas.append((q, temp_a(i, s), temp_b(b_tbase, j, s),
                                          sa_op, sb_in_base + s, osel))
            # read_order: distinct temps in first-use order over the MFMA stream.
            read_order = []; pos = {}
            for (q, at, bt, _sa, _sb, _o) in mfmas:
                for tt in (at, bt):
                    if tt not in pos:
                        pos[tt] = len(read_order); read_order.append(tt)
            total = len(read_order)         # == ntmp
            # ds_read addr/offset for each temp: which region/sub/tile -> base input + i*ts.
            def read_line(tt):
                if tt < t_bl:               # region A
                    rel = tt - t_a; i = rel // n_sub; s = rel % n_sub
                    return f"ds_read_b128 ${tt}, ${i_ab + s} offset:{i * ts_a}"
                elif tt < t_br:             # region BL
                    rel = tt - t_bl; j = rel // n_sub; s = rel % n_sub
                    return f"ds_read_b128 ${tt}, ${i_blb + s} offset:{j * ts_b}"
                else:                       # region BR
                    rel = tt - t_br; j = rel // n_sub; s = rel % n_sub
                    return f"ds_read_b128 ${tt}, ${i_brb + s} offset:{j * ts_b}"

            # first-use mfma index per temp (when it's first consumed).
            first_use = {}
            for midx, (q, at, bt, _sa, _sb, _o) in enumerate(mfmas):
                for tt in (at, bt):
                    if tt not in first_use:
                        first_use[tt] = midx
            # aiter-style 1:1: spread the ds_read EVENLY across the MFMA stream (issue each
            # read D mfma-steps before its first use) instead of all-upfront (which bursts
            # the LDS read port). issue_at[ri] monotonic non-decreasing in read_order.
            _D = int(__import__("os").environ.get("FP4_DSRD", "8"))   # read-ahead depth
            issue_at = []
            _prev = 0
            for ri, tt in enumerate(read_order):
                ia = max(0, first_use[tt] - _D)
                ia = max(ia, _prev)            # keep issue order == read_order
                issue_at.append(ia); _prev = ia
            # map mfma idx -> list of read_order indices to issue right before it
            sched = {}
            for ri, ia in enumerate(issue_at):
                sched.setdefault(ia, []).append(ri)

            glines = []
            for t in range(ng):
                o = i_g + t * 4
                glines.append(f"s_mov_b32 m0, ${o}\nbuffer_load_dwordx4 ${o+1}, ${o+2}, ${o+3} offen lds")
            gap = max(len(mfmas) // max(ng, 1), 1)
            out = []
            gi = 0; last_lgkm = None; issued = 0
            for idx, (q, at, bt, sa_op, sb_in, osel) in enumerate(mfmas):
                for ri in sched.get(idx, []):           # interleave reads scheduled here
                    out.append(read_line(read_order[ri])); issued += 1
                needed_done = max(pos[at], pos[bt]) + 1  # reads (in order) that must complete
                need = issued - needed_done              # outstanding reads allowed
                if need < 0:
                    need = 0
                if need != last_lgkm:
                    out.append(f"s_waitcnt lgkmcnt({need})"); last_lgkm = need
                out.append(f"v_mfma_scale_f32_16x16x128_f8f6f4 ${q}, ${at}, ${bt}, ${q}, "
                           f"${sa_op}, ${sb_in} {osel} cbsz:4 blgp:4")
                if gi < ng and idx % gap == 0:
                    out.append(glines[gi]); gi += 1
            while gi < ng:
                out.append(glines[gi]); gi += 1

            _ac = "=a" if _agpr else "=v"
            cons = ",".join(
                [_ac] * NT + ["=&v"] * ntmp                       # outputs: accs + read temps
                + ["v"] * (3 * n_sub)                             # a/bl/br base addrs
                + ["v"] * (2 * n_sub + 2 * n_sub)                 # sa(2*ns) + sbl(ns) + sbr(ns)
                + ["s", "v", "s", "s"] * ng                       # g2s
                + [str(q) for q in range(NT)])                    # tied accs
            st = "!llvm.struct<(" + ", ".join(
                ["vector<4xf32>"] * NT + ["vector<4xi32>"] * ntmp) + ")>"
            _cache[key] = ("\n".join(out), cons, st, NT, ntmp)
        asm, cons, st, NT, ntmp = _cache[key]
        ins = []
        for s in range_constexpr(n_sub):
            ins.append(_raw(a_base[s]))
        for s in range_constexpr(n_sub):
            ins.append(_raw(bl_base[s]))
        for s in range_constexpr(n_sub):
            ins.append(_raw(br_base[s]))
        for s in range_constexpr(n_sub):
            for gg in range_constexpr(2):
                ins.append(_raw(sa[s][gg]))
        for sc in (sbl, sbr):
            for s in range_constexpr(n_sub):
                ins.append(_raw(sc[s]))
        for (lds_p, voff, rsrc, soff) in g2s:
            ins.append(_raw(lds_p)); ins.append(_raw(voff)); ins.append(_raw(rsrc)); ins.append(_raw(soff))
        for q in range_constexpr(nq):
            ins.append(_raw(cL[q]))
        for q in range_constexpr(nq):
            ins.append(_raw(cR[q]))
        r = _llvm.inline_asm(ir.Type.parse(st), ins, asm, cons, has_side_effects=True)
        o = [Vec(_llvm.extractvalue(ir.Type.parse("vector<4xf32>"), r, [q])) for q in range_constexpr(nq * 2)]
        return o[:nq], o[nq:]

    def call_packed_asm2_g2s_dsr_pf(self, a_cur, an_base, bl_base, br_base, ts_a, ts_b,
                                    cL, cR, sa, sbl, sbr, n_sub, g2s, _cache={}):
        """HYBRID v3 (aiter cross-iter double-buffer): MFMA consumes a_cur (A frags read
        LAST iter -> their latency hid by last iter's MFMA = true cross-iter prefetch, vs
        compiler's implicit 1-iter hoist). Meanwhile THIS asm reads (a) NEXT iter's A into
        an_temp (output -> caller carries to next as a_cur) and (b) THIS iter's B into
        b_temp, both spread 1:1 across the MFMA stream. lgkmcnt gates only B (a_cur ready,
        an_next not needed this iter). Returns (accL, accR, a_next_frags). a_cur = nested
        [i][s] i32x4 input frags; an_base/bl/br_base = n_sub per-sub tile-0 LDS addrs."""
        assert self.packed
        nta, ntb = self.n_tiles_a, self.n_tiles_b
        nq = nta * ntb; NT = 2 * nq
        na, nb = nta * n_sub, ntb * n_sub
        ntmp = na + 2 * nb                 # outputs carried/internal: a_next(na) + B(2nb)
        ng = len(g2s)
        _agpr = __import__("os").environ.get("FP4_AGPR", "1") == "1"
        _D = int(__import__("os").environ.get("FP4_DSRD", "8"))
        key = (nta, ntb, n_sub, ng, ts_a, ts_b, _agpr, _D)
        if key not in _cache:
            t_an = NT                       # a_next temps (output, carried)
            t_bl = t_an + na; t_br = t_bl + nb
            in0 = NT + ntmp
            i_acur = in0                    # a_cur input frags (na)
            i_anb = i_acur + na             # a_next base addrs (n_sub)
            i_blb = i_anb + n_sub; i_brb = i_blb + n_sub
            i_sa = i_brb + n_sub; i_sbl = i_sa + 2 * n_sub; i_sbr = i_sbl + n_sub
            i_g = i_sbr + n_sub

            def acur(i, s): return i_acur + i * n_sub + s          # input frag operand
            def tb(base, j, s): return base + j * n_sub + s        # B temp output
            def tan(i, s): return t_an + i * n_sub + s             # a_next temp output

            # MFMA stream: a=a_cur(input), b=B temp(read this iter).
            mfmas = []
            for (sl, b_tbase, sb_in_base) in ((0, t_bl, i_sbl), (1, t_br, i_sbr)):
                for s in range(n_sub):
                    for i in range(nta):
                        for j in range(ntb):
                            q = sl * nq + i * ntb + j
                            oa, ob = i % 4, j
                            sa_op = i_sa + s * 2 + i // 4
                            osel = (f"op_sel:[{oa & 1},{ob & 1},0] op_sel_hi:[{(oa >> 1) & 1},{(ob >> 1) & 1},0]")
                            mfmas.append((q, acur(i, s), tb(b_tbase, j, s), sa_op, sb_in_base + s, osel))
            # B read_order (consumption order) + first-use mfma idx.
            b_order = []; bpos = {}; b_first = {}
            for midx, (q, ac, bt, _sa, _sb, _o) in enumerate(mfmas):
                if bt not in bpos:
                    bpos[bt] = len(b_order); b_order.append(bt); b_first[bt] = midx
            # a_next reads: spread evenly across the stream (filler, not gated this iter).
            an_temps = [tan(i, s) for i in range(nta) for s in range(n_sub)]
            an_gap = max(len(mfmas) // max(len(an_temps), 1), 1)

            def b_read_line(tt):
                rel0 = tt - t_bl
                if tt < t_br:
                    j = rel0 // n_sub; s = rel0 % n_sub
                    return f"ds_read_b128 ${tt}, ${i_blb + s} offset:{j * ts_b}"
                rel = tt - t_br; j = rel // n_sub; s = rel % n_sub
                return f"ds_read_b128 ${tt}, ${i_brb + s} offset:{j * ts_b}"

            def an_read_line(tt):
                rel = tt - t_an; i = rel // n_sub; s = rel % n_sub
                return f"ds_read_b128 ${tt}, ${i_anb + s} offset:{i * ts_a}"

            # schedule B reads D ahead of first use; track global issue order (B+an).
            b_issue_at = []; _prev = 0
            for tt in b_order:
                ia = max(0, b_first[tt] - _D); ia = max(ia, _prev); b_issue_at.append(ia); _prev = ia
            bsched = {}
            for k2, ia in enumerate(b_issue_at):
                bsched.setdefault(ia, []).append(b_order[k2])

            glines = []
            for t in range(ng):
                o = i_g + t * 4
                glines.append(f"s_mov_b32 m0, ${o}\nbuffer_load_dwordx4 ${o+1}, ${o+2}, ${o+3} offen lds")
            gap = max(len(mfmas) // max(ng, 1), 1)
            out = []
            gi = 0; ai = 0; last_lgkm = None; issued = 0
            gpos = {}   # global issue position of each B temp
            for idx, (q, ac, bt, sa_op, sb_in, osel) in enumerate(mfmas):
                for tt in bsched.get(idx, []):              # B reads (gated)
                    out.append(b_read_line(tt)); gpos[tt] = issued; issued += 1
                if ai < len(an_temps) and idx % an_gap == 0:  # a_next reads (filler)
                    out.append(an_read_line(an_temps[ai])); issued += 1; ai += 1
                need = issued - (gpos[bt] + 1)              # outstanding allowed (gate B)
                if need < 0:
                    need = 0
                if need != last_lgkm:
                    out.append(f"s_waitcnt lgkmcnt({need})"); last_lgkm = need
                out.append(f"v_mfma_scale_f32_16x16x128_f8f6f4 ${q}, ${ac}, ${bt}, ${q}, "
                           f"${sa_op}, ${sb_in} {osel} cbsz:4 blgp:4")
                if gi < ng and idx % gap == 0:
                    out.append(glines[gi]); gi += 1
            while ai < len(an_temps):
                out.append(an_read_line(an_temps[ai])); ai += 1
            while gi < ng:
                out.append(glines[gi]); gi += 1

            _ac = "=a" if _agpr else "=v"
            cons = ",".join(
                [_ac] * NT + ["=&v"] * ntmp
                + ["v"] * na                                  # a_cur input frags
                + ["v"] * (3 * n_sub)                         # an/bl/br base addrs
                + ["v"] * (2 * n_sub + 2 * n_sub)             # scales
                + ["s", "v", "s", "s"] * ng
                + [str(q) for q in range(NT)])
            st = "!llvm.struct<(" + ", ".join(
                ["vector<4xf32>"] * NT + ["vector<4xi32>"] * ntmp) + ")>"
            _cache[key] = ("\n".join(out), cons, st, NT, ntmp, na)
        asm, cons, st, NT, ntmp, na = _cache[key]
        ins = []
        for i in range_constexpr(nta):
            for s in range_constexpr(n_sub):
                ins.append(_raw(a_cur[i][s]))
        for s in range_constexpr(n_sub):
            ins.append(_raw(an_base[s]))
        for s in range_constexpr(n_sub):
            ins.append(_raw(bl_base[s]))
        for s in range_constexpr(n_sub):
            ins.append(_raw(br_base[s]))
        for s in range_constexpr(n_sub):
            for gg in range_constexpr(2):
                ins.append(_raw(sa[s][gg]))
        for sc in (sbl, sbr):
            for s in range_constexpr(n_sub):
                ins.append(_raw(sc[s]))
        for (lds_p, voff, rsrc, soff) in g2s:
            ins.append(_raw(lds_p)); ins.append(_raw(voff)); ins.append(_raw(rsrc)); ins.append(_raw(soff))
        for q in range_constexpr(nq):
            ins.append(_raw(cL[q]))
        for q in range_constexpr(nq):
            ins.append(_raw(cR[q]))
        r = _llvm.inline_asm(ir.Type.parse(st), ins, asm, cons, has_side_effects=True)
        o = [Vec(_llvm.extractvalue(ir.Type.parse("vector<4xf32>"), r, [q])) for q in range_constexpr(NT)]
        # a_next frags (carried): extract the na i32x4 temps after the NT accs.
        an = [_llvm.extractvalue(ir.Type.parse("vector<4xi32>"), r, [NT + t]) for t in range_constexpr(na)]
        an_nested = [[Vec(an[i * n_sub + s]) for s in range_constexpr(n_sub)] for i in range_constexpr(nta)]
        return o[:nq], o[nq:], an_nested

    def call_mxfp4_wholeloop(self, a_base, bl_base, br_base, ts_a, ts_b,
                             abase, blbase, brbase, gl_a, gl_b, rsrc_a, rsrc_b,
                             kstep, scv, cL, cR, n_sub, nsa, nsb, nval, soff0, soff0_bl, soff0_br,
                             sc_rb, sc_gb, sc_rsa, sc_rsb, sc_voff, sc_soff0, sca_rb=None, sca_gb=None, sca_voff=None, _cache={}):
        """WHOLE-LOOP bare-asm (aiter structure, perf-validated _asm9 ~5722): the ENTIRE
        K-loop is ONE inline-asm hw-loop -> no per-iter FlyDSL boundary / operand passing
        (the +16% lever, NOT double-buffer). 2 LDS buffers (buf0/buf1) ping-pong, unroll-2
        (divides KI=112). Each phase: ds_read operands (single reg buffer) + 128 MFMA
        (L+R, n_sub) const scale + G2S buffer_load_lds refill (2-ahead, advancing gmem
        soffset) + s_barrier. CONSTANT scale (perf-first per user; real scale added after).
        a_base[b][s]/bl_base[b][s]/br_base[b][s]: ds_read LDS addrs (b=buf0/1). abase[b]/
        blbase[b]/brbase[b]: per-wave G2S LDS dest base SGPR (m0 = base+step*NW*1024).
        gl_a[st]/gl_b[st]: per-lane gmem voffsets. rsrc_a/b: buffer resources. Returns
        (accL,accR). Caller passes a_soff/b_soff init as the LAST tied in/out via cL... no:
        soffsets are created inside via the advancing SGPRs (passed as kstep-scaled)."""
        assert self.packed
        nta, ntb = self.n_tiles_a, self.n_tiles_b
        nq = nta * ntb; NT = 2 * nq
        na, nb = nta * n_sub, ntb * n_sub
        ntmp = na + 2 * nb
        _NWc = 4   # n_waves (4-wave kernel)
        nbuf = len(a_base)   # A pool size (ring 4 = refill-OTHER 2-ahead)
        nbuf_b = len(bl_base)  # B pool size (may exceed A pool: asymmetric 2A+4B aiter-style)
        _nscbuf = 4 if int(__import__("os").environ.get("FP4_SCDWX4_2A", "0")) else nbuf_b  # scale LDS pool (2-ahead needs 4)
        _ASYM = (nbuf != nbuf_b)  # asymmetric: A shallow (2), B deep (4) -> deep B g2s in-flight
        _RING = (nbuf == 4 and not _ASYM)  # 4-buffer ring: register-prefetch + refill-OTHER 2-ahead
        _WLDB = _RING or int(__import__("os").environ.get("FP4_WLDB", "0"))  # register double-buffer
        _BPREF = int(__import__("os").environ.get("FP4_BPREF", "0"))  # ASYMMETRIC: A single-set + B double-set
        # FP4_SUBSTREAM: aiter-style SUB-granular streaming. 2 reg sub-sets (1 sub each = 136 VGPR,
        # fits at BK256 unlike NSET=2 tile-level 272V) + LDS ping-pong(nbuf buffers, each n_sub subs)
        # + g2s front-loaded + deep vmcnt. Gives BK256 intensity WITH hidden ds_read. nbuf>=2.
        _SS = int(__import__("os").environ.get("FP4_SUBSTREAM", "0"))
        _SUB = _SS or _ASYM   # sub-granular layout (2 reg sub-sets): SS (sym) or ASYM (2A+4B pools)
        _SCA2 = int(__import__("os").environ.get("FP4_SC_A2", "0"))  # A-scale combined dwordx2 (g0,g1)
        _TRB8 = int(__import__("os").environ.get("FP4_TRB8", "0"))  # gluon scale path: lane-major LDS + ds_read_b64 pairs (halve scale reads). needs PIN+PINSC.
        _TRB8BASE = int(__import__("os").environ.get("FP4_PINBASE", "8"))
        _SCVGPR = int(__import__("os").environ.get("FP4_SC_VGPR", "0"))  # VGPR-direct scale: buffer_load_dwordx4 lane-contig -> t_sc (no LDS/ds_write/staging). needs PIN+PINSC.
        _SCV2AHEAD = int(__import__("os").environ.get("FP4_SCV_2AHEAD", "0"))  # 2-ahead scale prefetch: load own set AFTER mfma (next-iter same phase) -> 2 vmcnt barriers (VMEM out-of-order drain @ WLV>0).
        _SCDWX4 = int(__import__("os").environ.get("FP4_SCDWX4", "0"))  # scale preshuffle(lane-contig) -> 2 buffer_load_dwordx4...lds (A,B packed regions) replace 8 dword g2s; ds_read_b32 back. LDS/lgkmcnt (det-safe). aiter g2s 8 vs FlyDSL 16 dword/256mfma.
        NSET = 2 if (_WLDB and not _SUB) else 1
        key = (nta, ntb, n_sub, nsa, nsb, ts_a, ts_b, nbuf, _WLDB, _RING, _BPREF, _SS, _SCA2, _TRB8, _SCVGPR, _SCV2AHEAD, _SCDWX4,
               int(__import__("os").environ.get("FP4_SUBSTREAM_VM", "10")),
               int(__import__("os").environ.get("FP4_SS_UNROLL", "0")),
               int(__import__("os").environ.get("FP4_PIN", "0")),
               int(__import__("os").environ.get("FP4_PINBASE", "8")),
               int(__import__("os").environ.get("FP4_INPLACE", "0")),
               int(__import__("os").environ.get("FP4_WLVMCN", "0")),
               int(__import__("os").environ.get("FP4_WLSYNC", "0")),
               __import__("os").environ.get("FP4_INPLACE_ALT", "1"),
               __import__("os").environ.get("FP4_INPLACE_ELGK", "0"),
               __import__("os").environ.get("FP4_INPLACE_DIAG", "0"),
               __import__("os").environ.get("FP4_INPLACE_SCOV", "0"),
               __import__("os").environ.get("FP4_INPLACE_1BAR", "0"),
               __import__("os").environ.get("FP4_FEWOP", "0"),
               __import__("os").environ.get("FP4_NSUBFOLD", "0"),
               __import__("os").environ.get("FP4_NF_ELGK", "0"),
               __import__("os").environ.get("FP4_NF_CONSEC", "1"),
               __import__("os").environ.get("FP4_PINBF", "0"))
        if key not in _cache:
            o_acc = list(range(NT))
            t_a = NT; t_bl = t_a + na; t_br = t_bl + nb        # ds_read temp outputs (set 0)
            nsct = 4 * n_sub                                    # scale temps: A-g0,A-g1,BL,BR x n_sub
            if _BPREF:
                # ASYMMETRIC: A single(t_a) + B double(set0 t_bl/t_br, set1 t_bl1/t_br1) + scale single.
                # 48 operand frags(16A+32B) fit VGPR(192)+256 AGPR; mfma reads ready A+B[cur],
                # cross-iter prefetch B[next] hides B's ds_read on natural emit_mm order (keeps 6772).
                t_bl1 = t_br + nb; t_br1 = t_bl1 + nb
                t_sc = t_br1 + nb
                ntmp2 = na + 4 * nb + nsct
                set_sz = ntmp + nsct
            elif _SUB:
                # SUB-STREAM: NSS=2 reg sub-sets, each ONE sub (nfs frags + 4 scales). Fits BK256
                # (2*(16*4+4)=136 VGPR) unlike NSET=2 tile-level (272). Accessors via ss_* below.
                nfs = nta + 2 * ntb                 # A(nta)+BL(ntb)+BR(ntb) frags per sub
                sub_sz = nfs + 4                     # + 4 scale groups (A-g0,A-g1,BL,BR)
                t_bl = NT + nta; t_br = NT + nta + ntb; t_sc = NT + nfs   # ss=0 bases
                set_sz = sub_sz
                ntmp2 = 2 * sub_sz                   # NSS=2 sub-sets
            else:
                t_sc = t_br + nb                                # scale temp base (set 0)
                # TRB8: + nsct staging VGPRs (buffer_load scale->VGPR, then ds_write lane-major
                # to LDS so it can be read back as ds_read_b64 pairs). t_stage = t_sc + nsct.
                t_stage = t_sc + nsct        # TRB8 staging / SCVGPR 2nd scale set (ping-pong)
                # SCVGPR: +nsct (2 sets ping-pong); SCV2AHEAD: +3*nsct (4 sets); SCDWX4: +nsct (VGPR stage for GR->ds_write)
                _scextra = (3 * nsct if (_SCVGPR and _SCV2AHEAD) else (nsct if (_TRB8 or _SCVGPR or _SCDWX4) else 0))
                set_sz = ntmp + nsct + _scextra  # temps per set
                ntmp2 = NSET * set_sz                          # set s temps at NT + s*set_sz + local
            o_cnt = NT + ntmp2                                  # =&s loop counter
            o_sa = o_cnt + 1; o_sbl = o_sa + 1; o_sbr = o_sbl + 1   # advancing gmem soffsets A/BL/BR
            o_ta = o_sbr + 1; o_tbl = o_ta + 1; o_tbr = o_tbl + 1   # buf1 (=+kstep) scratch soffsets
            o_sca = [o_tbr + 1 + g for g in range(4)]           # 4 scale soffsets (A-g0,A-g1,BL,BR)
            o_sct = o_sca[3] + 1                                # scale s=1 scratch (+256)
            nout = o_sct + 1
            # scale temp accessors (group: 0=A-g0,1=A-g1,2=BL,3=BR; slot=grp*n_sub+s)
            # _scb[0] = SCVGPR ping-pong scale-set base (0 or nsct), set per emit_inplace phase.
            _scb = [0]
            _scrdbuf = [0]   # SCDWX4_2A: scale LDS read buffer index (4-buf 2-ahead cycling), set per phase
            def sa_t(s, g): return (t_sc + s * 2 + g) if _SCA2 else (t_sc + _scb[0] + g * n_sub + s)
            def sbl_t(s): return t_sc + _scb[0] + 2 * n_sub + s
            def sbr_t(s): return t_sc + _scb[0] + 3 * n_sub + s
            # inputs (after outputs):
            i = nout
            i_ab = [[i + b * n_sub + s for s in range(n_sub)] for b in range(nbuf)]; i += nbuf * n_sub   # A ds_read base (nbuf_a)
            i_blb = [[i + b * n_sub + s for s in range(n_sub)] for b in range(nbuf_b)]; i += nbuf_b * n_sub  # B pool nbuf_b
            i_brb = [[i + b * n_sub + s for s in range(n_sub)] for b in range(nbuf_b)]; i += nbuf_b * n_sub
            i_g_ab = [i + b for b in range(nbuf)]; i += nbuf   # g2s A LDS dest base (sgpr)
            i_g_blb = [i + b for b in range(nbuf_b)]; i += nbuf_b
            i_g_brb = [i + b for b in range(nbuf_b)]; i += nbuf_b
            i_gla = [i + s for s in range(nsa)]; i += nsa       # gmem voffsets A
            i_glb = [i + s for s in range(nsb)]; i += nsb       # gmem voffsets B
            i_rsa = i; i += 1; i_rsb = i; i += 1                # rsrc
            i_kstep = i; i += 1
            i_sc = i; i += 1                                    # (legacy const scale, unused now)
            i_nval = i; i += 1
            i_sa0 = i; i += 1; i_sbl0 = i; i += 1; i_sbr0 = i; i += 1   # soffset inits A/BL/BR (region base k=0)
            i_scrb = [i + b for b in range(_nscbuf)]; i += _nscbuf   # scale LDS read base (_nscbuf; 4 for 2-ahead)
            i_scgb = [i + b for b in range(_nscbuf)]; i += _nscbuf   # scale LDS g2s dest base
            i_scrsa = i; i += 1; i_scrsb = i; i += 1            # scale rsrc (A_scale, B_scale)
            i_scvoff = i; i += 1                                # scale per-lane gmem voffset (lane*4)
            i_sca0 = [i + g for g in range(4)]; i += 4          # scale soffset inits (A-g0,A-g1,BL,BR)
            # a2 inputs only consume register slots when enabled (else INPLACE reg-alloc shifts)
            if _SCA2:
                i_scra2 = [i + b for b in range(nbuf_b)]; i += nbuf_b  # a2 A-scale LDS read base (lane*8)
                i_scvf2 = i; i += 1                             # a2 A-scale gmem voffset (lane*8)
            else:
                i_scra2 = i_scrb; i_scvf2 = i_scvoff           # unused dummies (no extra slots)

            def emit_ds(buf, off=0):
                r = []
                for ii in range(nta):
                    for s in range(n_sub):
                        r.append(f"ds_read_b128 ${t_a + ii*n_sub+s+off}, ${i_ab[buf][s]} offset:{ii*ts_a}")
                for ji in range(ntb):
                    for s in range(n_sub):
                        r.append(f"ds_read_b128 ${t_bl + ji*n_sub+s+off}, ${i_blb[buf][s]} offset:{ji*ts_b}")
                for ji in range(ntb):
                    for s in range(n_sub):
                        r.append(f"ds_read_b128 ${t_br + ji*n_sub+s+off}, ${i_brb[buf][s]} offset:{ji*ts_b}")
                # scales (4 groups x n_sub) from SC_lds[buf], b32, slot grp*n_sub+s at off slot*256
                if _SCDWX4:
                    _rd = int(__import__("os").environ.get("FP4_SCDWX4_RD", "128"))
                    if _rd == 32:
                        for slot in range(nsct):
                            offb = slot * 4 if slot < 2 * n_sub else 1024 + (slot - 2 * n_sub) * 4
                            r.append(f"ds_read_b32 ${t_sc + slot + off}, ${i_scrb[buf]} offset:{offb}")
                        return r
                    # lane-packed LDS: ds_read_b128 (4 scales/op) A@i_scrb, B@+1024. writes pinned v[pb:pb+3].
                    _pinb = int(__import__("os").environ.get("FP4_PINBASE", "8"))
                    for grp in (0, 2 * n_sub):
                        pb = _pinb + grp + off
                        offb = 0 if grp == 0 else 1024
                        r.append(f"ds_read_b128 v[{pb}:{pb+3}], ${i_scrb[buf]} offset:{offb}")
                    return r
                if _SCVGPR:
                    return r   # VGPR-direct: scales loaded by emit_sc_vgpr in loop, not from LDS
                if _TRB8:
                    # lane-major: slot at byte offset slot*4 (matches ds_write g2s + ds_line read)
                    for slot in range(nsct):
                        r.append(f"ds_read_b32 ${t_sc + slot+off}, ${i_scrb[buf]} offset:{slot*4}")
                elif _SCA2:
                    # a2: A scales (g0,g1) from SCA region (lane*8 + region*4); b64 to LDS pair
                    # is not expressible via single-reg inline-asm operands -> 2 b32. B as b32.
                    for s in range(n_sub):
                        for region in (0, 1):
                            r.append(f"ds_read_b32 ${t_sc + s*2 + region+off}, ${i_scra2[buf]} offset:{s*512 + region*4}")
                    for slot in range(2 * n_sub, nsct):
                        r.append(f"ds_read_b32 ${t_sc + slot+off}, ${i_scrb[buf]} offset:{slot*256}")
                else:
                    for slot in range(nsct):
                        r.append(f"ds_read_b32 ${t_sc + slot+off}, ${i_scrb[buf]} offset:{slot*256}")
                return r

            _MMORD = int(__import__("os").environ.get("FP4_MMORD", "0"))  # mfma emission order (bank-conflict probe)
            def emit_mm(off=0):
                r = []
                def one(sl, tb, sbfn, s, ii, ji):
                    q = sl * nq + ii * ntb + ji
                    oa, ob = ii % 4, ji
                    osel = f"op_sel:[{oa&1},{ob&1},0] op_sel_hi:[{(oa>>1)&1},{(ob>>1)&1},0]"
                    return (f"v_mfma_scale_f32_16x16x128_f8f6f4 ${q}, ${t_a+ii*n_sub+s+off}, "
                            f"${tb+ji*n_sub+s+off}, ${q}, ${sa_t(s, ii//4)+off}, ${sbfn(s)+off} {osel} cbsz:4 blgp:4")
                for (sl, tb, sbfn) in ((0, t_bl, sbl_t), (1, t_br, sbr_t)):
                    for s in range(n_sub):
                        if _MMORD == 1:    # ji outer, ii inner (different acc sequence)
                            for ji in range(ntb):
                                for ii in range(nta):
                                    r.append(one(sl, tb, sbfn, s, ii, ji))
                        elif _MMORD == 2:  # ii stride-2 interleave (split bank groups)
                            for ii in list(range(0, nta, 2)) + list(range(1, nta, 2)):
                                for ji in range(ntb):
                                    r.append(one(sl, tb, sbfn, s, ii, ji))
                        elif _MMORD == 3:  # aiter 2x2 register-block: 2 A-rows interleaved at 2-B granularity
                            for ii2 in range(0, nta, 2):
                                for ji2 in range(0, ntb, 2):
                                    for di in range(2):
                                        for dj in range(2):
                                            r.append(one(sl, tb, sbfn, s, ii2 + di, ji2 + dj))
                        elif _MMORD == 4:  # 4 A-chains x 2-B block (more independent mfma chains)
                            for ii4 in range(0, nta, 4):
                                for ji2 in range(0, ntb, 2):
                                    for di in range(4):
                                        for dj in range(2):
                                            r.append(one(sl, tb, sbfn, s, ii4 + di, ji2 + dj))
                        elif _MMORD == 5:  # 2 A-chains x 4-B block (B-inner wider)
                            for ii2 in range(0, nta, 2):
                                for di in range(2):
                                    for ji in range(ntb):
                                        r.append(one(sl, tb, sbfn, s, ii2 + di, ji))
                        else:
                            for ii in range(nta):
                                for ji in range(ntb):
                                    r.append(one(sl, tb, sbfn, s, ii, ji))
                return r

            def emit_g2s(buf, sa_op, sbl_op, sbr_op):
                r = []
                for st in range(nsa):
                    r.append(f"s_add_u32 m0, ${i_g_ab[buf]}, {st*_NWc*1024}\n"
                             f"buffer_load_dwordx4 ${i_gla[st]}, ${i_rsa}, ${sa_op} offen lds")
                for st in range(nsb):
                    r.append(f"s_add_u32 m0, ${i_g_blb[buf]}, {st*_NWc*1024}\n"
                             f"buffer_load_dwordx4 ${i_glb[st]}, ${i_rsb}, ${sbl_op} offen lds")
                for st in range(nsb):
                    r.append(f"s_add_u32 m0, ${i_g_brb[buf]}, {st*_NWc*1024}\n"
                             f"buffer_load_dwordx4 ${i_glb[st]}, ${i_rsb}, ${sbr_op} offen lds")
                return r

            def emit_scale_g2s(buf, base_extra):
                # refill SC_lds[buf] with 4 groups x n_sub scale dwords. soffset =
                # o_sca[grp] + base_extra + s*256 (base_extra picks the K-iter: 0 = k=2t+2,
                # _scstep = k=2t+3 for the refill-OTHER scheme).
                r = []
                if _SCDWX4 and int(__import__("os").environ.get("FP4_SCDWX4_DIRECT", "0")):
                    # DIRECT: combined gmem->LDS (no staging/ds_write). vmcnt, drained by _ipenda. ds_line=b128.
                    # A lane-packed @ m0=i_scgb (write=m0+voffset(lane*16)); B @ +1024.
                    # FP4_SCDWX4_NARROW: 8 dword-lds (write per dword, NON-racy; dwordx4-lds WRITE races!).
                    # else 2 dwordx4-lds (fewer but racy). NARROW: m0=base+s*4, voffset=lane*16, gmem offset:s*4.
                    # (base_extra always 0 in the in-loop/prologue calls -> o_sca[0]/[2] used directly)
                    if int(__import__("os").environ.get("FP4_SCDWX4_DW2", "0")):
                        # dwordx2-lds VALIDITY+TRACKING test (user's a2 dwordx2). 4 pairs (A p0/p1, B p0/p1).
                        # gmem pair p = 2 contig dwords @ lane*16+p*8; LDS pair region @ m0=base+p*512.
                        for p in range(2 * n_sub // 2):
                            r.append(f"s_add_u32 m0, ${i_scgb[buf]}, {p*512}\n"
                                     f"buffer_load_dwordx2 ${i_scvoff}, ${i_scrsa}, ${o_sca[0]} offen offset:{p*8} lds")
                        for p in range(2 * n_sub // 2):
                            r.append(f"s_add_u32 m0, ${i_scgb[buf]}, {1024 + p*512}\n"
                                     f"buffer_load_dwordx2 ${i_scvoff}, ${i_scrsb}, ${o_sca[2]} offen offset:{p*8} lds")
                    elif int(__import__("os").environ.get("FP4_SCDWX4_NARROW", "1")):
                        for s in range(2 * n_sub):
                            r.append(f"s_add_u32 m0, ${i_scgb[buf]}, {s*4}\n"
                                     f"buffer_load_dword ${i_scvoff}, ${i_scrsa}, ${o_sca[0]} offen offset:{s*4} lds")
                        for s in range(2 * n_sub):
                            r.append(f"s_add_u32 m0, ${i_scgb[buf]}, {1024 + s*4}\n"
                                     f"buffer_load_dword ${i_scvoff}, ${i_scrsb}, ${o_sca[2]} offen offset:{s*4} lds")
                    else:
                        r.append(f"s_add_u32 m0, ${i_scgb[buf]}, 0\n"
                                 f"buffer_load_dwordx4 ${i_scvoff}, ${i_scrsa}, ${o_sca[0]} offen lds")
                        r.append(f"s_add_u32 m0, ${i_scgb[buf]}, 1024\n"
                                 f"buffer_load_dwordx4 ${i_scvoff}, ${i_scrsb}, ${o_sca[2]} offen lds")
                    return r
                if _SCDWX4:
                    # gluon GR (only here): buffer_load_dwordx4 preshuffled scale -> VGPR stage (4 contig/op).
                    # NO vmcnt(0) here -> overlaps mfma. LW (ds_write -> lane-packed LDS) emitted AFTER
                    # _ipenda (GR landed by phase vmcnt) via emit_scale_lw. LR = ds_read_b128 in ds_line.
                    _pinb = int(__import__("os").environ.get("FP4_PINBASE", "8"))
                    stA = _pinb + nsct; stB = _pinb + nsct + 2 * n_sub
                    soffA, soffB = o_sca[0], o_sca[2]
                    if base_extra != 0:
                        r.append(f"s_add_u32 ${o_sct}, ${o_sca[0]}, {base_extra}"); soffA = o_sct
                    r.append(f"buffer_load_dwordx4 v[{stA}:{stA+3}], ${i_scvoff}, ${i_scrsa}, ${soffA} offen")
                    if base_extra != 0:
                        r.append(f"s_add_u32 ${o_sct}, ${o_sca[2]}, {base_extra}"); soffB = o_sct
                    r.append(f"buffer_load_dwordx4 v[{stB}:{stB+3}], ${i_scvoff}, ${i_scrsb}, ${soffB} offen")
                    return r
                if _TRB8:
                    # gluon scale path: buffer_load scale -> VGPR stage, vmcnt(0), then ds_write
                    # to lane-major LDS (i_scrb[buf]=wave_region+lane*nsct, slot at +slot*4) so it
                    # reads back as ds_read_b64 pairs. gmem voff=lane*4. vmcnt(0) MUST precede the
                    # ds_write (else it reads un-landed gmem load -> nan).
                    # lane_contig gmem read (= SCVGPR's CORRECT read): A 2*n_sub dwords from
                    # rsrc_a@o_sca[0], B 2*n_sub from rsrc_b@o_sca[2], lane stride 16 (i_scvoff=lane*16),
                    # dword slot at offset slot*4 (slot=g*n_sub+s = sa_t/sb_t idx). Then ds_write
                    # lane-major LDS -> ds_read_b64 pairs (det0). Fixes the packed/per-dword value bug.
                    for slot in range(2 * n_sub):
                        r.append(f"buffer_load_dword ${t_stage+slot}, ${i_scvoff}, ${i_scrsa}, ${o_sca[0]} offen offset:{slot*4}")
                    for slot in range(2 * n_sub):
                        r.append(f"buffer_load_dword ${t_stage+2*n_sub+slot}, ${i_scvoff}, ${i_scrsb}, ${o_sca[2]} offen offset:{slot*4}")
                    r.append("s_waitcnt vmcnt(0)")
                    for slot in range(4 * n_sub):
                        r.append(f"ds_write_b32 ${i_scrb[buf]}, ${t_stage+slot} offset:{slot*4}")
                    return r
                if _SCA2:
                    # A (a2): buffer_load...lds is DWORD-ONLY on CDNA4 (no dwordx2 to LDS),
                    # so emit 2 dword loads (region0,1) into the interleaved a2 LDS layout
                    # (lane*8 + region*4). gmem region1 = +1 dword (offset:4). soffset same.
                    for s in range(n_sub):
                        tot = base_extra * 2 + s * 512
                        soff = o_sca[0]
                        if tot != 0:
                            r.append(f"s_add_u32 ${o_sct}, ${o_sca[0]}, {tot}")
                            soff = o_sct
                        for region in (0, 1):
                            r.append(f"s_add_u32 m0, ${i_scgb[buf]}, {s*512 + region*4}\n"
                                     f"buffer_load_dword ${i_scvf2}, ${i_scrsa}, ${soff} offen offset:{region*4} lds")
                    # B (grp 2,3): unchanged dword, slots 2*n_sub.. (after A's 2*n_sub slots)
                    for grp in (2, 3):
                        for s in range(n_sub):
                            slot = grp * n_sub + s
                            tot = base_extra + s * 256
                            if tot == 0:
                                r.append(f"s_add_u32 m0, ${i_scgb[buf]}, {slot*256}\n"
                                         f"buffer_load_dword ${i_scvoff}, ${i_scrsb}, ${o_sca[grp]} offen lds")
                            else:
                                r.append(f"s_add_u32 ${o_sct}, ${o_sca[grp]}, {tot}\n"
                                         f"s_add_u32 m0, ${i_scgb[buf]}, {slot*256}\n"
                                         f"buffer_load_dword ${i_scvoff}, ${i_scrsb}, ${o_sct} offen lds")
                    return r
                # TRB8: lane-major LDS (lane*nsct + slot) so a lane's nsct scale dwords are
                # contiguous -> read back as ds_read_b64 pairs. m0 picks the slot within the
                # lane region (slot*4); caller sets i_scvoff = lane*nsct*4 (the lane's region).
                _mslot = 4 if _TRB8 else 256
                for grp in range(4):
                    rsrc = i_scrsa if grp < 2 else i_scrsb
                    for s in range(n_sub):
                        slot = grp * n_sub + s
                        tot = base_extra + s * 256
                        if tot == 0:
                            r.append(f"s_add_u32 m0, ${i_scgb[buf]}, {slot*_mslot}\n"
                                     f"buffer_load_dword ${i_scvoff}, ${rsrc}, ${o_sca[grp]} offen lds")
                        else:
                            r.append(f"s_add_u32 ${o_sct}, ${o_sca[grp]}, {tot}\n"
                                     f"s_add_u32 m0, ${i_scgb[buf]}, {slot*_mslot}\n"
                                     f"buffer_load_dword ${i_scvoff}, ${rsrc}, ${o_sct} offen lds")
                return r

            def emit_scale_lw(buf):
                # SCDWX4 LW: ds_write staging (GR'd, landed) -> lane-packed LDS buf (A@i_scrb, B@+1024).
                # FP4_SCDWX4_LW: 128=ds_write_b128 (1/op); 32=4x ds_write_b32 (diag: isolate b128 vs addr).
                _pinb = int(__import__("os").environ.get("FP4_PINBASE", "8"))
                stA = _pinb + nsct; stB = _pinb + nsct + 2 * n_sub
                _lw = int(__import__("os").environ.get("FP4_SCDWX4_LW", "128"))
                if _lw == 32:
                    r2 = []
                    for j in range(2 * n_sub):
                        r2.append(f"ds_write_b32 ${i_scrb[buf]}, v{stA + j} offset:{j*4}")
                        r2.append(f"ds_write_b32 ${i_scrb[buf]}, v{stB + j} offset:{1024 + j*4}")
                    return r2
                return [f"ds_write_b128 ${i_scrb[buf]}, v[{stA}:{stA+3}]",
                        f"ds_write_b128 ${i_scrb[buf]}, v[{stB}:{stB+3}] offset:1024"]

            def interleave(mm, g2s):
                gap = max(len(mm) // max(len(g2s), 1), 1)
                out = []; gi = 0
                for idx, m in enumerate(mm):
                    out.append(m)
                    if gi < len(g2s) and idx % gap == 0:
                        out.append(g2s[gi]); gi += 1
                while gi < len(g2s):
                    out.append(g2s[gi]); gi += 1
                return out

            # ds_read line per temp (for the interleaved/staggered scheduling, FP4_WLDSR)
            def ds_line(buf, tt):
                if tt < t_bl:
                    rel = tt - t_a; ii = rel // n_sub; s = rel % n_sub
                    return f"ds_read_b128 ${tt}, ${i_ab[buf][s]} offset:{ii*ts_a}"
                if tt < t_br:
                    rel = tt - t_bl; ji = rel // n_sub; s = rel % n_sub
                    return f"ds_read_b128 ${tt}, ${i_blb[buf][s]} offset:{ji*ts_b}"
                if tt < t_sc:
                    rel = tt - t_br; ji = rel // n_sub; s = rel % n_sub
                    return f"ds_read_b128 ${tt}, ${i_brb[buf][s]} offset:{ji*ts_b}"
                slot = tt - t_sc
                if _SCDWX4:
                    if int(__import__("os").environ.get("FP4_SCDWX4_BFLIP","0")): buf = 1 - buf
                    if slot >= nsct:
                        return ""
                    _rbuf = _scrdbuf[0] if int(__import__("os").environ.get("FP4_SCDWX4_2A", "0")) else buf  # 2A: 4-buf 2-ahead read idx
                    _rd = int(__import__("os").environ.get("FP4_SCDWX4_RD", "128"))
                    if _rd == 32:
                        off = slot * 4 if slot < 2 * n_sub else 1024 + (slot - 2 * n_sub) * 4
                        return f"ds_read_b32 ${tt}, ${i_scrb[_rbuf]} offset:{off}"
                    if slot % (2 * n_sub) != 0:
                        return ""
                    _pinbase = int(__import__("os").environ.get("FP4_PINBASE", "8"))
                    pb = _pinbase + slot
                    off = 0 if slot < 2 * n_sub else 1024
                    return f"ds_read_b128 v[{pb}:{pb+3}], ${i_scrb[_rbuf]} offset:{off}"
                if _SCVGPR:
                    return ""   # VGPR-direct: scale temps filled by emit_sc_vgpr, not from LDS
                if _TRB8:
                    # lane-major: slot at byte offset slot*4.
                    if int(__import__("os").environ.get("FP4_TRB8_B64", "0")) and slot % 2 == 0:
                        pb = _TRB8BASE + slot
                        return f"ds_read_b64 v[{pb}:{pb+1}], ${i_scrb[buf]} offset:{slot*4}"
                    if int(__import__("os").environ.get("FP4_TRB8_B64", "0")):
                        return ""
                    return f"ds_read_b32 ${tt}, ${i_scrb[buf]} offset:{slot*4}"
                return f"ds_read_b32 ${tt}, ${i_scrb[buf]} offset:{slot*256}"

            _WLDSR = int(__import__("os").environ.get("FP4_WLDSR", "0"))
            _WLD = int(__import__("os").environ.get("FP4_WLDSRD", "8"))   # read-ahead depth

            def emit_phase_dsr(buf, g2sl):
                # interleave ds_read (operands+scales) INTO the mfma stream (staggered
                # lgkmcnt) instead of read-all-upfront -> reads overlap the mfma (aiter).
                # mfma metadata (q, a_t, b_t, sa_t, sb_t) + the temps each consumes.
                mlist = []
                for (sl, tb, sbfn) in ((0, t_bl, sbl_t), (1, t_br, sbr_t)):
                    for s in range(n_sub):
                        for ii in range(nta):
                            for ji in range(ntb):
                                q = sl * nq + ii * ntb + ji
                                oa, ob = ii % 4, ji
                                osel = f"op_sel:[{oa&1},{ob&1},0] op_sel_hi:[{(oa>>1)&1},{(ob>>1)&1},0]"
                                at = t_a + ii * n_sub + s; bt = tb + ji * n_sub + s
                                sat = sa_t(s, ii // 4); sbt = sbfn(s)
                                mline = (f"v_mfma_scale_f32_16x16x128_f8f6f4 ${q}, ${at}, ${bt}, ${q}, "
                                         f"${sat}, ${sbt} {osel} cbsz:4 blgp:4")
                                mlist.append((mline, [at, bt, sat, sbt]))
                # read_order = temps in first-use order
                ro = []; pos = {}; first = {}
                for mi, (_ml, deps) in enumerate(mlist):
                    for tt in deps:
                        if tt not in pos:
                            pos[tt] = len(ro); ro.append(tt); first[tt] = mi
                total = len(ro)
                # issue_at: D ahead of first use, monotone
                iat = {}; prev = 0
                for tt in ro:
                    a = max(0, first[tt] - _WLD); a = max(a, prev); iat[tt] = a; prev = a
                sched = {}
                for tt in ro:
                    sched.setdefault(iat[tt], []).append(tt)
                # g2s (refill-SAME) must start ONLY after all ds_read of this buffer issued,
                # else g2s clobbers a buffer still being read. Spread over the tail mfma.
                g2s_start = (max(iat.values()) + 1) if iat else 0
                tail = max(len(mlist) - g2s_start, 1)
                gap = max(tail // max(len(g2sl), 1), 1)
                out = []; issued = 0; last = None; gi = 0
                for mi, (ml, deps) in enumerate(mlist):
                    for tt in sched.get(mi, []):
                        out.append(ds_line(buf, tt)); issued += 1
                    need = issued - (max(pos[d] for d in deps) + 1)
                    if need < 0: need = 0
                    if need != last:
                        out.append(f"s_waitcnt lgkmcnt({need})"); last = need
                    out.append(ml)
                    if gi < len(g2sl) and mi >= g2s_start and (mi - g2s_start) % gap == 0:
                        out.append(g2sl[gi]); gi += 1
                while gi < len(g2sl):
                    out.append(g2sl[gi]); gi += 1
                return out

            _INPLACE = int(__import__("os").environ.get("FP4_INPLACE", "0"))

            def emit_inplace(nxt_buf, g2sl, side="A"):
                # NEXT-K in-place refill (aiter mechanism, VGPR-feasible, NSET=1, 2 LDS bufs):
                # emit mfma; after the refill-SIDE operand's LAST use, ds_read its NEXT-k tile from
                # nxt_buf into the freed reg -> read latency overlaps remaining mfma. The other side
                # (used till end) is end-drained. Emission ORDER picks which side frees progressively:
                # side="A" -> ii-outer (A progressive); side="B" -> ji-outer (B progressive). Accs are
                # order-independent (sum) so reorder is safe. Alternating side per phase overlaps BOTH.
                # build cell list. side="DIAG": traverse the ii x (2*ntb) grid in diagonal order so
                # BOTH A[ii] and B[col] free progressively (each refilled mid-loop). side="A"/"B":
                # one side progressive (other end-drained). Accs order-independent -> reorder safe.
                cells = []   # (ii, sl, ji)
                _mmo = int(__import__("os").environ.get("FP4_MMORD", "0"))
                _mm3 = _mmo in (3, 4, 5)
                if side == "DIAG" and _mm3:
                    # BLOCKED DIAGONAL: both-progressive free (DIAG hide) + N-chain throughput +
                    # wider-bn restores A-operand reuse (raises DIAG-削 ceiling toward natural 7555).
                    # block (bm A-rows x bn cols): MMORD 3=2x2, 4=4x2 (deeper A-reuse), 5=2x4 (wider).
                    bm, bn = {3: (2, 2), 4: (4, 2), 5: (2, 4)}[_mmo]
                    ncol = 2 * ntb; nib = nta // bm; ncb = ncol // bn
                    # FP4_SINNER: put s (K-sub) INNERMOST so the same acc's n_sub MFMA are
                    # consecutive (gluon/amdgcnas pattern: acc stays in the MFMA PE, full-rate
                    # accumulation). Default (s outer) spreads same-acc by a full diagonal -> bubbles.
                    _sinner = int(__import__("os").environ.get("FP4_SINNER", "0"))
                    _accd16 = int(__import__("os").environ.get("FP4_ACC_DIST16", "0"))
                    if _accd16:
                        # aiter cross-bank cadence: partition the nta*(2*ntb) distinct accs
                        # into col-groups of 2 (=> 8 ii x 2 col = 16 distinct accs/group).
                        # Within a group cycle all 16 distinct accs (bank-strided ii order)
                        # BEFORE repeating for the next K-sub s -> each acc recurs at dist 16
                        # (vs s-outer dist 64), leaving 15 independent mfma to hide each
                        # operand ds_read (aiter's mem-latency-hiding). Accs are order-free.
                        ncol = 2 * ntb
                        iorder = [0, 2, 4, 6, 1, 3, 5, 7][:nta] if nta == 8 else list(range(nta))
                        cg2 = int(__import__("os").environ.get("FP4_ACC_DIST16_NCOL", "2"))
                        for cg in range(0, ncol, cg2):
                            for s in range(n_sub):
                                for c2 in range(cg2):
                                    col = cg + c2
                                    if col >= ncol:
                                        continue
                                    for ii in iorder:
                                        cells.append((ii, col // ntb, col % ntb, s))
                    elif _sinner:
                        for D in range(nib + ncb - 1):
                            for iib in range(nib):
                                cb = D - iib
                                if 0 <= cb < ncb:
                                    for di in range(bm):
                                        for dj in range(bn):
                                            for s in range(n_sub):
                                                ii = iib * bm + di; col = cb * bn + dj
                                                cells.append((ii, col // ntb, col % ntb, s))
                    else:
                        for s in range(n_sub):
                            for D in range(nib + ncb - 1):
                                for iib in range(nib):
                                    cb = D - iib
                                    if 0 <= cb < ncb:
                                        for di in range(bm):
                                            for dj in range(bn):
                                                ii = iib * bm + di; col = cb * bn + dj
                                                cells.append((ii, col // ntb, col % ntb, s))
                elif side == "DIAG":
                    ncol = 2 * ntb
                    for s in range(n_sub):
                        for d in range(nta + ncol - 1):
                            for ii in range(nta):
                                col = d - ii
                                if 0 <= col < ncol:
                                    cells.append((ii, col // ntb, col % ntb, s))
                elif int(__import__("os").environ.get("FP4_MMORD", "0")) == 3:
                    # aiter 2x2 register-block (throughput) + A-progressive (side A hides A): block ii,ji
                    for sl in (0, 1):
                        for s in range(n_sub):
                            for ii2 in range(0, nta, 2):
                                for ji2 in range(0, ntb, 2):
                                    for di in range(2):
                                        for dj in range(2):
                                            cells.append((ii2 + di, sl, ji2 + dj, s))
                else:
                    for sl in (0, 1):
                        for s in range(n_sub):
                            outer = range(nta) if side == "A" else range(ntb)
                            inner = range(ntb) if side == "A" else range(nta)
                            for o_ in outer:
                                for i_ in inner:
                                    ii, ji = (o_, i_) if side == "A" else (i_, o_)
                                    cells.append((ii, sl, ji, s))
                mlist = []
                for (ii, sl, ji, s) in cells:
                    tb = t_bl if sl == 0 else t_br
                    sbfn = sbl_t if sl == 0 else sbr_t
                    q = sl * nq + ii * ntb + ji
                    oa, ob = ii % 4, ji
                    osel = f"op_sel:[{oa&1},{ob&1},0] op_sel_hi:[{(oa>>1)&1},{(ob>>1)&1},0]"
                    at = t_a + ii * n_sub + s; bt = tb + ji * n_sub + s
                    sat = sa_t(s, ii // 4); sbt = sbfn(s)
                    # FP4_FEWOP: speed-probe — force mfma to READ from few distinct operand regs
                    # (garbage output) to test if distinct-operand count caps mfma throughput.
                    _fop = int(__import__("os").environ.get("FP4_FEWOP", "0"))
                    a_r, b_r = (t_a + (at - t_a) % (_fop * n_sub), tb + (bt - tb) % (_fop * n_sub)) if _fop else (at, bt)
                    mline = (f"v_mfma_scale_f32_16x16x128_f8f6f4 ${q}, ${a_r}, ${b_r}, ${q}, "
                             f"${sat}, ${sbt} {osel} cbsz:4 blgp:4")
                    mlist.append((mline, at, bt, sat, sbt))
                last = {}
                for mi, (_ml, at, bt, sat, sbt) in enumerate(mlist):
                    last[at] = mi; last[bt] = mi; last[sat] = mi; last[sbt] = mi
                # which temps refill mid-loop (after last use): DIAG=both operands(+scales if SCOV),
                # else one side. _scov: also overlap scale-temps (t_sc..) instead of end-draining.
                _scov = int(__import__("os").environ.get("FP4_INPLACE_SCOV", "0"))
                schi = (NT + set_sz) if (side == "DIAG" and _scov) else t_sc
                if side == "DIAG":
                    mid = set(t for t in last if t_a <= t < schi)
                elif side == "A":
                    mid = set(t for t in last if t_a <= t < t_bl)
                else:
                    mid = set(t for t in last if t_bl <= t < t_sc)
                _ipnodsr = int(__import__("os").environ.get("FP4_WLNODSR", "0"))  # ceiling probe: skip refills
                # aiter-style pacing (replicate aiter .s: 17 s_nop + 16 fine lgkmcnt vs fly 0 + 1
                # coarse lgkmcnt(0)/phase). Both gates only ADD s_nop / s_waitcnt; never reorder mfma
                # or drop/move a ds_read. FP4_FINELGK=0 FP4_MFMANOP=0 reproduces current bytes exactly.
                _finelgk = int(__import__("os").environ.get("FP4_FINELGK", "0"))
                _mfmanop = int(__import__("os").environ.get("FP4_MFMANOP", "0"))     # s_nop COUNT per gap
                _nopgap = int(__import__("os").environ.get("FP4_MFMANOP_GAP", "8"))  # mfma between nops
                if int(__import__("os").environ.get("FP4_WLNOG2S", "0")):  # ceiling probe: skip g2s
                    g2sl = []
                out = []; gi = 0; refilled = set()
                # FP4_WLRING_GFRAC: aiter g2s front-load (g2s in first gfrac of mfma, tail clear->lands)
                _gfr_ip = float(__import__("os").environ.get("FP4_WLRING_GFRAC", "0"))
                # FP4_INPLACE_GLATE: g2s in LAST glate-fraction of mfma (DIAG refills are early/progressive
                # -> g2s in the sparse tail avoids collision; opposite of GFRAC front-load).
                _glate = float(__import__("os").environ.get("FP4_INPLACE_GLATE", "0"))
                _gstart = 0
                if _glate > 0 and g2sl:
                    _gstart = max(len(mlist) - int(len(mlist) * _glate), 0)
                    _glim = len(mlist); ngap = max((len(mlist) - _gstart) // max(len(g2sl), 1), 1)
                elif _gfr_ip > 0 and g2sl:
                    _glim = max(int(len(mlist) * _gfr_ip), len(g2sl))
                    ngap = max(_glim // max(len(g2sl), 1), 1)
                else:
                    _glim = len(mlist); ngap = max(len(mlist) // max(len(g2sl), 1), 1)
                # FP4_INPLACE_GAVOID: place g2s ONLY at mfma slots with NO ds_read refill (avoid
                # shadow over-subscription where refills cluster). Spread g2s over the free slots.
                _gavoid = int(__import__("os").environ.get("FP4_INPLACE_GAVOID", "0"))
                if _gavoid and not _ipnodsr and g2sl:
                    # precompute refill slots
                    _rfslot = set()
                    _rf = set()
                    for mi, (ml, at, bt, sat, sbt) in enumerate(mlist):
                        for rt in (at, bt, sat, sbt):
                            if rt in mid and last[rt] == mi and rt not in _rf:
                                _rfslot.add(mi); _rf.add(rt)
                    _free = [mi for mi in range(len(mlist)) if mi not in _rfslot]
                    _fgap = max(len(_free) // max(len(g2sl), 1), 1)
                    _gset = {}
                    for _k, _fi in enumerate(_free):
                        if (_k % _fgap == 0) and len(_gset) < len(g2sl):
                            _gset[_fi] = len(_gset)
                # FP4_EVENSPREAD: instead of emitting each refill at its operand's last-use
                # (clusters where last-uses bunch), DEFER refills to a ready-queue and emit them
                # EVENLY (1 per `gap` mfma) so the ds_read fill the v_mfma_scale operand-latency
                # bubbles uniformly (gluon/amdgcnas pattern: 64 ds_read spread over 256 mfma).
                _evenspread = int(__import__("os").environ.get("FP4_EVENSPREAD", "0"))
                if _evenspread and not _ipnodsr:
                    ready_at = {}
                    seen_rt = set()
                    for mi, (ml, at, bt, sat, sbt) in enumerate(mlist):
                        for rt in (at, bt, sat, sbt):
                            if rt in mid and last[rt] == mi and rt not in seen_rt:
                                ready_at.setdefault(mi, []).append(rt); seen_rt.add(rt)
                    nref = len(seen_rt); rgap = max(len(mlist) // max(nref, 1), 1)
                    rq = []
                    for mi, (ml, at, bt, sat, sbt) in enumerate(mlist):
                        out.append(ml)
                        if mi in ready_at: rq.extend(ready_at[mi])
                        if rq and (mi % rgap == 0):
                            rt = rq.pop(0); out.append(ds_line(nxt_buf, rt)); refilled.add(rt)
                        if gi < len(g2sl) and mi >= _gstart and (mi - _gstart) % ngap == 0 and mi < _glim:
                            out.append(g2sl[gi]); gi += 1
                    while rq:
                        rt = rq.pop(0); out.append(ds_line(nxt_buf, rt)); refilled.add(rt)
                    while gi < len(g2sl):
                        out.append(g2sl[gi]); gi += 1
                else:
                    # FINELGK precompute: count operand+scale refills emitted THIS phase (consumed
                    # next phase). Terminal _ipend lgkmcnt(0) at phase boundary is the det0 backstop;
                    # FINELGK only inserts EARLIER, looser partial drains (monotone non-increasing
                    # toward 0) -> can only wait MORE, never less -> correctness preserved by construction.
                    _nref = 0
                    if _finelgk and not _ipnodsr:
                        _seen = set()
                        for _mi, (_ml, _at, _bt, _sat, _sbt) in enumerate(mlist):
                            for _rt in (_at, _bt, _sat, _sbt):
                                if _rt in mid and last[_rt] == _mi and _rt not in _seen:
                                    _seen.add(_rt); _nref += 1
                    _fl_depth = int(__import__("os").environ.get("FP4_FINELGK_D", "4"))  # in-flight refills to keep
                    _emit = 0; _lastlgk = None; _nopc = 0
                    for mi, (ml, at, bt, sat, sbt) in enumerate(mlist):
                        out.append(ml)
                        if _mfmanop:
                            _nopc += 1
                            if _nopc >= _nopgap:
                                out.append(f"s_nop {_mfmanop}"); _nopc = 0
                        if not _ipnodsr:
                            for rt in (at, bt, sat, sbt):
                                if rt in mid and last[rt] == mi and rt not in refilled:
                                    out.append(ds_line(nxt_buf, rt)); refilled.add(rt); _emit += 1
                        if _finelgk and not _ipnodsr and _emit:
                            _tgt = max(0, _emit - _fl_depth)
                            _rem = min(15, _nref - _tgt)   # gfx950 lgkmcnt field max = 15
                            if _rem <= 15 and (_lastlgk is None or _rem < _lastlgk):
                                out.append(f"s_waitcnt lgkmcnt({_rem})"); _lastlgk = _rem
                        if _gavoid and not _ipnodsr and g2sl:
                            if mi in _gset and gi < len(g2sl):
                                out.append(g2sl[gi]); gi += 1
                        elif gi < len(g2sl) and mi >= _gstart and (mi - _gstart) % ngap == 0 and mi < _glim:
                            out.append(g2sl[gi]); gi += 1
                    while gi < len(g2sl):
                        out.append(g2sl[gi]); gi += 1
                # end drain: refill the other-side operand + all scale-temps (used till end).
                if not _ipnodsr:
                    for tt in range(t_a, NT + set_sz):
                        if tt not in refilled:
                            out.append(ds_line(nxt_buf, tt))
                return out

            _NSUBFOLD = int(__import__("os").environ.get("FP4_NSUBFOLD", "0"))

            def emit_nsubfold(cur_buf, nxt_buf, g2sl):
                # n_sub-FOLD (candidate to capture FEWOP 8-op ceiling without BLOCK_K's 2x barrier):
                # mfma read only the s=0-slot operand regs (8 distinct) for BOTH K-subs; refill those
                # 8 regs in-place between sub0 and sub1 (s1 data, same buf) and to next phase (s0,
                # nxt_buf). Scales stay un-folded (prologue/refill loads all nsct). 1 barrier/phase.
                _ipnodsr = int(__import__("os").environ.get("FP4_WLNODSR", "0"))
                # FP4_NF_CONSEC: fold to CONSECUTIVE regs (t_a..t_a+nta-1) instead of stride-n_sub.
                # FEWOP showed consecutive 8-reg reads = 5623 vs stride-2 = 5011 (read-port/bank).
                _consec = int(__import__("os").environ.get("FP4_NF_CONSEC", "0"))
                if _consec:
                    a_r = lambda ii: t_a + ii
                    bl_r = lambda ji: t_bl + ji
                    br_r = lambda ji: t_br + ji
                else:
                    a_r = lambda ii: t_a + ii * n_sub
                    bl_r = lambda ji: t_bl + ji * n_sub
                    br_r = lambda ji: t_br + ji * n_sub
                # (reg, tile, kind) for the 8 folded operand regs; kind 0=A,1=BL,2=BR
                specs = ([(a_r(ii), ii, 0) for ii in range(nta)]
                         + [(bl_r(ji), ji, 1) for ji in range(ntb)]
                         + [(br_r(ji), ji, 2) for ji in range(ntb)])
                reg2spec = {r: (r, tl, k) for (r, tl, k) in specs}
                def rd(reg, buf, s, tile, kind):
                    base = (i_ab[buf][s] if kind == 0 else i_blb[buf][s] if kind == 1 else i_brb[buf][s])
                    off = tile * (ts_a if kind == 0 else ts_b)
                    return f"ds_read_b128 ${reg}, ${base} offset:{off}"
                def mm_sub(s):
                    r = []
                    for sl, breg, sbfn in ((0, bl_r, sbl_t), (1, br_r, sbr_t)):
                        for ii in range(nta):
                            for ji in range(ntb):
                                q = sl * nq + ii * ntb + ji
                                oa, ob = ii % 4, ji
                                osel = f"op_sel:[{oa&1},{ob&1},0] op_sel_hi:[{(oa>>1)&1},{(ob>>1)&1},0]"
                                at = a_r(ii); bt = breg(ji)
                                sat = sa_t(s, ii // 4); sbt = sbfn(s)
                                ml = (f"v_mfma_scale_f32_16x16x128_f8f6f4 ${q}, ${at}, ${bt}, ${q}, "
                                      f"${sat}, ${sbt} {osel} cbsz:4 blgp:4")
                                r.append((ml, at, bt))
                    return r
                def subiter(mm, rbuf, rs, g2s_tail, scale_refill):
                    last = {}
                    for mi, (_m, at, bt) in enumerate(mm):
                        last[at] = mi; last[bt] = mi
                    o = []; gi = 0; refilled = set()
                    ngap = max(len(mm) // max(len(g2s_tail), 1), 1)
                    for mi, (ml, at, bt) in enumerate(mm):
                        o.append(ml)
                        if not _ipnodsr:
                            for rt in (at, bt):
                                if last[rt] == mi and rt not in refilled:
                                    _, tl, k = reg2spec[rt]
                                    o.append(rd(rt, rbuf, rs, tl, k)); refilled.add(rt)
                        if gi < len(g2s_tail) and mi % ngap == 0:
                            o.append(g2s_tail[gi]); gi += 1
                    while gi < len(g2s_tail):
                        o.append(g2s_tail[gi]); gi += 1
                    if scale_refill and not _ipnodsr:   # cross-phase: refresh all scales from nxt_buf
                        for slot in range(nsct):
                            o.append(f"ds_read_b32 ${t_sc + slot}, ${i_scrb[rbuf]} offset:{slot*256}")
                    return o
                out = subiter(mm_sub(0), cur_buf, 1, [], False)       # sub0: refill ops <- cur s1
                out.append(f"s_waitcnt lgkmcnt({int(__import__('os').environ.get('FP4_NF_ELGK','0'))})")
                out += subiter(mm_sub(1), nxt_buf, 0, g2sl, True)     # sub1: refill ops+scales <- nxt s0
                return out

            # FP4_WLSYNC: sync mode. 0=vmcnt(0)+barrier (safe). 1=vmcnt(N)+barrier (keep N
            # g2s in flight). 2=barrier only (no vmcnt drain, speed-ceiling probe).
            _WLS = int(__import__("os").environ.get("FP4_WLSYNC", "0"))
            _WLV = int(__import__("os").environ.get("FP4_WLVMCN", "0"))
            _NODSR = int(__import__("os").environ.get("FP4_WLNODSR", "0"))  # ceiling probe: skip operand ds_read (garbage)
            # FP4_WLBARNOP: s_nop after s_barrier (aiter: 2x s_nop 0 post-barrier pre-mfma, settle barrier).
            _BARNOP = int(__import__("os").environ.get("FP4_WLBARNOP", "0"))
            _bnop = "\ns_nop 0" * _BARNOP
            if _WLS == 2:
                _endph = _endpha = "s_barrier" + _bnop
            elif _WLS == 3:
                # 1 barrier/body: phase A vmcnt-only (no barrier), phase B vmcnt+barrier.
                _endpha = f"s_waitcnt vmcnt({_WLV})"
                _endph = f"s_waitcnt vmcnt({_WLV})\ns_barrier" + _bnop
            elif _WLS == 1:
                _endph = _endpha = f"s_waitcnt vmcnt({_WLV})\ns_barrier" + _bnop
            else:
                _endph = _endpha = "s_waitcnt vmcnt(0)\ns_barrier" + _bnop
            # ===== SUB-STREAM (FP4_SUBSTREAM / ASYM) emit helpers =====
            if _SUB:
                def ss_a(ss, ii): return NT + ss * sub_sz + ii
                def ss_bl(ss, ji): return NT + ss * sub_sz + nta + ji
                def ss_br(ss, ji): return NT + ss * sub_sz + nta + ntb + ji
                def ss_sc(ss, g): return NT + ss * sub_sz + nfs + g   # g: 0=A-g0,1=A-g1,2=BL,3=BR
                _ssmm3 = int(__import__("os").environ.get("FP4_MMORD", "0")) == 3
                def emit_mm_ss(ss):
                    r = []
                    def _emit(sl, bget, scg, ii, ji):
                        q = sl * nq + ii * ntb + ji
                        oa, ob = ii % 4, ji
                        osel = f"op_sel:[{oa&1},{ob&1},0] op_sel_hi:[{(oa>>1)&1},{(ob>>1)&1},0]"
                        r.append(f"v_mfma_scale_f32_16x16x128_f8f6f4 ${q}, ${ss_a(ss,ii)}, "
                                 f"${bget(ss,ji)}, ${q}, ${ss_sc(ss,ii//4)}, ${ss_sc(ss,scg)} {osel} cbsz:4 blgp:4")
                    for (sl, bget, scg) in ((0, ss_bl, 2), (1, ss_br, 3)):
                        if _ssmm3:   # aiter 2x2 register-block
                            for ii2 in range(0, nta, 2):
                                for ji2 in range(0, ntb, 2):
                                    for di in range(2):
                                        for dj in range(2):
                                            _emit(sl, bget, scg, ii2 + di, ji2 + dj)
                        else:
                            for ii in range(nta):
                                for ji in range(ntb):
                                    _emit(sl, bget, scg, ii, ji)
                    return r
                def emit_ds_ss(ss, buf, sib, b_buf=None):  # reg-read sub: A from a-pool buf, B/scale from b_buf
                    bb = buf if b_buf is None else b_buf
                    r = []
                    for ii in range(nta):
                        r.append(f"ds_read_b128 ${ss_a(ss,ii)}, ${i_ab[buf][sib]} offset:{ii*ts_a}")
                    for ji in range(ntb):
                        r.append(f"ds_read_b128 ${ss_bl(ss,ji)}, ${i_blb[bb][sib]} offset:{ji*ts_b}")
                    for ji in range(ntb):
                        r.append(f"ds_read_b128 ${ss_br(ss,ji)}, ${i_brb[bb][sib]} offset:{ji*ts_b}")
                    for g in range(4):
                        r.append(f"ds_read_b32 ${ss_sc(ss,g)}, ${i_scrb[bb]} offset:{(g*n_sub+sib)*256}")
                    return r
                # split g2s: A-pool refill (A frags only) vs B-pool refill (BL+BR+scale)
                def emit_g2s_a(a_buf):
                    r = []
                    for st in range(nsa):
                        r.append(f"s_add_u32 m0, ${i_g_ab[a_buf]}, {st*_NWc*1024}\n"
                                 f"buffer_load_dwordx4 ${i_gla[st]}, ${i_rsa}, ${o_sa} offen lds")
                    return r
                def emit_g2s_b(b_buf):
                    r = []
                    for st in range(nsb):
                        r.append(f"s_add_u32 m0, ${i_g_blb[b_buf]}, {st*_NWc*1024}\n"
                                 f"buffer_load_dwordx4 ${i_glb[st]}, ${i_rsb}, ${o_sbl} offen lds")
                    for st in range(nsb):
                        r.append(f"s_add_u32 m0, ${i_g_brb[b_buf]}, {st*_NWc*1024}\n"
                                 f"buffer_load_dwordx4 ${i_glb[st]}, ${i_rsb}, ${o_sbr} offen lds")
                    return r + emit_scale_g2s(b_buf, 0)
            L = [f"s_mov_b32 ${o_cnt}, 0",
                 f"s_mov_b32 ${o_sa}, ${i_sa0}", f"s_mov_b32 ${o_sbl}, ${i_sbl0}",
                 f"s_mov_b32 ${o_sbr}, ${i_sbr0}"]
            for g in range(4):
                L.append(f"s_mov_b32 ${o_sca[g]}, ${i_sca0[g]}")
            # SCVGPR scale prefetch: emit_sc_vgpr(tb) loads this kk's 8 scale dwords lane-contig
            # DIRECT to t_sc[tb:tb+8] via 2 buffer_load_dwordx4 (A->[tb:tb+4], B->[tb+4:tb+8]).
            # NO vmcnt here (in flight; drained by the phase-end vmcnt) -> overlaps mfma, scale free.
            # ping-pong sets (tb=0 / tb=nsct) so mfma reads the set loaded last phase. needs PIN+PINSC.
            _pbsc = _TRB8BASE
            _scv_dwx4 = int(__import__("os").environ.get("FP4_SCV_DWX4", "1"))  # 1=dwordx4 literal; 0=4 tracked dword
            def emit_sc_vgpr(tb):
                p = _pbsc + tb
                if _scv_dwx4:
                    return [f"buffer_load_dwordx4 v[{p}:{p+3}], ${i_scvoff}, ${i_scrsa}, ${o_sca[0]} offen",
                            f"buffer_load_dwordx4 v[{p+4}:{p+7}], ${i_scvoff}, ${i_scrsb}, ${o_sca[2]} offen"]
                # tracked dword: write ${t_sc+...} operands (LLVM sees def -> no literal-reg race)
                r = [f"buffer_load_dword ${t_sc+tb+i}, ${i_scvoff}, ${i_scrsa}, ${o_sca[0]} offen offset:{i*4}" for i in range(4)]
                r += [f"buffer_load_dword ${t_sc+tb+4+i}, ${i_scvoff}, ${i_scrsb}, ${o_sca[2]} offen offset:{i*4}" for i in range(4)]
                return r
            _scvstep = 64 * (2 * n_sub) * 4   # lane-contig kk stride in bytes
            def _scv_adv():
                return [f"s_add_u32 ${o_sca[0]}, ${o_sca[0]}, {_scvstep}",
                        f"s_add_u32 ${o_sca[2]}, ${o_sca[2]}, {_scvstep}"]
            # SCDWX4: fill scale LDS buf0(k=0)+buf1(k=1) lane-packed (2 dwordx4 each) BEFORE emit_ds
            # reads scale. o_sca starts at k=0; advance _scvstep/buffer -> ends at k=2 (= loop entry).
            if _SCDWX4:
                _direct_p = int(__import__("os").environ.get("FP4_SCDWX4_DIRECT", "0"))
                for _b in range(2):
                    L += emit_scale_g2s(_b, 0)               # DIRECT: dwordx4-lds->LDS; staging: GR->staging
                    if not _direct_p:
                        L.append("s_waitcnt vmcnt(0)")       # GR land
                        L += emit_scale_lw(_b)               # LW staging -> buf _b
                    L += [f"s_add_u32 ${o_sca[0]}, ${o_sca[0]}, {_scvstep}",
                          f"s_add_u32 ${o_sca[2]}, ${o_sca[2]}, {_scvstep}"]
                L.append("s_waitcnt vmcnt(0) lgkmcnt(0)")    # DIRECT: g2s(vmcnt) land; staging: LW(lgkmcnt) land. before emit_ds LR
            if _TRB8:
                # TRB8 prologue: prefill 2 scale LDS bufs (lane-major) + advance o_sca to kk=2,
                # matching the 2-buffer prefetch depth (else loop-entry mfma reads un-filled scale LDS
                # + wrong-K o_sca -> global scale-K shift). emit_scale_g2s does load->staging->ds_write.
                _trb8pf = int(__import__("os").environ.get("FP4_TRB8_PREFILL", "2"))
                for _b in range(_trb8pf):
                    L += emit_scale_g2s(_b % 2, 0)
                    L += [f"s_add_u32 ${o_sca[0]}, ${o_sca[0]}, {_scvstep}",
                          f"s_add_u32 ${o_sca[2]}, ${o_sca[2]}, {_scvstep}"]
                L.append("s_waitcnt vmcnt(0) lgkmcnt(0)")
                if int(__import__("os").environ.get("FP4_TRB8_PFBAR", "0")): L.append("s_barrier")
            # register double-buffer prologue: read buf0 (k=0) into set0 before the loop.
            # (BPREF: emit_ds(0,0) fills A + B-set0 + scale = k0 = phase-A entry state)
            if _SUB:
                L += emit_ds_ss(0, 0, 0); L.append("s_waitcnt lgkmcnt(0)")  # sub0 of buf0 -> ss-set0
            elif _WLDB or _INPLACE or _BPREF:
                L += emit_ds(0, 0); L.append("s_waitcnt lgkmcnt(0)")
            if _SCVGPR:
                if _SCV2AHEAD:
                    # 4-set double-buffer (A0=0,A1=nsct,B0=2nsct,B1=3nsct), unroll-by-2 par.
                    # prologue fills only i=0: A0<-kk0, B0<-kk1. A1/B1 filled in-body (2-ahead).
                    L += emit_sc_vgpr(0) + _scv_adv()          # A0 <- kk0 (phase A i=0)
                    L += emit_sc_vgpr(2 * nsct) + _scv_adv()   # B0 <- kk1 (phase B i=0)
                    L += ["s_waitcnt vmcnt(0)"]
                else:
                    L += emit_sc_vgpr(0) + _scv_adv() + ["s_waitcnt vmcnt(0)"]  # setA = phase A iter0 scales
            L.append("1:")
            _scstep = n_sub * 256
            if _ASYM:
                # ASYMMETRIC 2A+4B sub-stream (BK128, n_sub=1): A pool nbuf(2), B pool nbuf_b(4).
                # A refill-SAME nbuf-ahead (2), B refill-SAME nbuf_b-ahead (4 -> deep B g2s in-flight).
                # unroll = lcm(nbuf,nbuf_b) sub-iters; reg double-buffer (2 sub-sets). vmcnt(_WLV) deep.
                import math as _math
                _runa = (nbuf * nbuf_b) // _math.gcd(nbuf, nbuf_b)   # lcm
                for u in range(_runa):
                    a_buf = u % nbuf; b_buf = u % nbuf_b
                    ss_cur = u % 2; ss_nxt = (u + 1) % 2
                    na_buf = (u + 1) % nbuf; nb_buf = (u + 1) % nbuf_b
                    dsl = emit_ds_ss(ss_nxt, na_buf, 0, b_buf=nb_buf)   # n_sub=1 -> sib=0
                    # g2s: refill A pool (a_buf, read nbuf-ahead) + B pool (b_buf, read nbuf_b-ahead)
                    g2sl = emit_g2s_a(a_buf) + emit_g2s_b(b_buf)
                    # aiter g2s front-load (FP4_WLRING_GFRAC); else interleave
                    _gfr = float(__import__("os").environ.get("FP4_WLRING_GFRAC", "0"))
                    _mm = emit_mm_ss(ss_cur)
                    if int(__import__("os").environ.get("FP4_ASYM_AITERILV", "0")) and g2sl:
                        # EXACT aiter schedule: g2s after mfma 4k (k<ng); ds_read[k] after 4k+1 (first ng),
                        # remaining ds_read after 4*ng+2j (tail every-2). Replicates aiter256.s exactly.
                        _ng = len(g2sl); _nd = len(dsl)
                        _gpos = {4 * k: k for k in range(_ng)}
                        _rpos = {}; _nf = min(_ng, _nd)
                        for k in range(_nf): _rpos[4 * k + 1] = k
                        _ts = 4 * _ng; _j = 0
                        for di in range(_nf, _nd): _rpos[_ts + 2 * _j] = di; _j += 1
                        _o = []
                        for _i, _m in enumerate(_mm):
                            _o.append(_m)
                            if _i in _rpos: _o.append(dsl[_rpos[_i]])
                            if _i in _gpos: _o.append(g2sl[_gpos[_i]])
                        # safety: append any unplaced (if mm shorter than positions)
                        _pd = set(_rpos.values()); _pg = set(_gpos.values())
                        for di in range(_nd):
                            if di not in _pd: _o.append(dsl[di])
                        for gi2 in range(_ng):
                            if gi2 not in _pg: _o.append(g2sl[gi2])
                        L += _o
                    elif _gfr > 0 and g2sl:
                        _n = len(_mm); _greg = max(int(_n * _gfr), len(g2sl)); _gg = max(_greg // len(g2sl), 1)
                        _dg = max(_n // max(len(dsl), 1), 1); _gi = _di = 0; _o = []
                        for _ix, _m in enumerate(_mm):
                            _o.append(_m)
                            if _di < len(dsl) and _ix % _dg == 0: _o.append(dsl[_di]); _di += 1
                            if _gi < len(g2sl) and _ix < _greg and _ix % _gg == 0: _o.append(g2sl[_gi]); _gi += 1
                        while _di < len(dsl): _o.append(dsl[_di]); _di += 1
                        while _gi < len(g2sl): _o.append(g2sl[_gi]); _gi += 1
                        L += _o
                    else:
                        L += interleave(_mm, dsl + g2sl)
                    _abf = int(__import__("os").environ.get("FP4_ASYM_BARFREQ", "0"))  # 0=last only, N=every N sub-iter
                    _sbar = (u == _runa - 1) or (int(__import__("os").environ.get("FP4_ASYM_ALLBAR", "0")) == 1) \
                            or (_abf > 0 and (u % _abf) == (_abf - 1))
                    L.append(f"s_waitcnt vmcnt({_WLV}) lgkmcnt(0)" + ("\ns_barrier" if _sbar else ""))
                    # advance soffsets: A every nbuf? no -- both A and B g2s advance their own pos per refill.
                    for _so in (o_sa, o_sbl, o_sbr):
                        L.append(f"s_add_u32 ${_so}, ${_so}, ${i_kstep}")
                    for g in range(4):
                        L.append(f"s_add_u32 ${o_sca[g]}, ${o_sca[g]}, {2*_scstep if (_SCA2 and g==0) else _scstep}")
            elif _SS:
                # SUB-STREAM: nbuf LDS buffers (each n_sub subs), 2 reg sub-sets stream the subs.
                # tile-iter ti: mfma buf[ti]'s n_sub subs (reg sub-double-buffer) + refill buf[ti]
                # AFTER its subs reg-read -> T_{g+nbuf} (read nbuf tile-iters later = deep vmcnt OK).
                # lgkmcnt(0)/sub-iter (reg-read ready for next sub mfma); vmcnt(_WLV) deep g2s; barrier/tile-iter.
                _ss1bar = int(__import__("os").environ.get("FP4_SS_1BAR", "1"))
                for ti in range(nbuf):
                    for sib in range(n_sub):
                        gu = ti * n_sub + sib
                        ss_cur = gu % 2; ss_nxt = (gu + 1) % 2
                        if sib < n_sub - 1:
                            dsl = emit_ds_ss(ss_nxt, ti, sib + 1)
                        else:
                            dsl = emit_ds_ss(ss_nxt, (ti + 1) % nbuf, 0)
                        g2sl = (emit_g2s(ti, o_sa, o_sbl, o_sbr) + emit_scale_g2s(ti, 0)) if sib == n_sub - 1 else []
                        # FP4_WLRING_GFRAC: aiter g2s front-load (first gfrac of mfma, tail g2s-free->lands)
                        _gfr = float(__import__("os").environ.get("FP4_WLRING_GFRAC", "0"))
                        if _gfr > 0 and g2sl:
                            _mm = emit_mm_ss(ss_cur); _n = len(_mm)
                            _greg = max(int(_n * _gfr), len(g2sl)); _gg = max(_greg // len(g2sl), 1)
                            _dg = max(_n // max(len(dsl), 1), 1); _gi = _di = 0; _o = []
                            for _ix, _m in enumerate(_mm):
                                _o.append(_m)
                                if _di < len(dsl) and _ix % _dg == 0: _o.append(dsl[_di]); _di += 1
                                if _gi < len(g2sl) and _ix < _greg and _ix % _gg == 0: _o.append(g2sl[_gi]); _gi += 1
                            while _di < len(dsl): _o.append(dsl[_di]); _di += 1
                            while _gi < len(g2sl): _o.append(g2sl[_gi]); _gi += 1
                            L += _o
                        else:
                            L += interleave(emit_mm_ss(ss_cur), dsl + g2sl)
                        _last = (sib == n_sub - 1)
                        _sbar = _last and ((ti == nbuf - 1) or _ss1bar == 0)
                        L.append(f"s_waitcnt vmcnt({_WLV}) lgkmcnt(0)" + ("\ns_barrier" if _sbar else ""))
                        if _last:
                            for _so in (o_sa, o_sbl, o_sbr):
                                L.append(f"s_add_u32 ${_so}, ${_so}, ${i_kstep}")
                            for g in range(4):
                                L.append(f"s_add_u32 ${o_sca[g]}, ${o_sca[g]}, {2*_scstep if (_SCA2 and g==0) else _scstep}")
            elif _RING:
                # 4-BUFFER RING (hide ds_read, NO vmcnt0 penalty): unroll-4, 2 register sets.
                # sub-iter u: mfma(set cur=u%2, =buf[u] reg-read last sub-iter) interleaved with
                # reg-read buf[(u+1)%4]->set (u+1)%2 (1-ahead, hide ds_read) + g2s refill
                # buf[(u+2)%4] (2-ahead refill-OTHER, relaxed). o_sa/o_sca track the 2-ahead refill
                # pos (advance +kstep/+_scstep PER SUB-ITER) -> gbuf refill uses them directly.
                # vmcnt(0)/sub-iter drains the prev sub-iter's g2s (=this rbuf), ~free (1 sub-iter
                # old, landed under the prev 128 mfmas) -> NOT the 2-buf same-phase vmcnt0 penalty.
                # FP4_WLRING_2A: TRUE 2-ahead g2s (buf g2s'd at u, reg-read at u+2) -> vmcnt(_WLV>0)
                # has 2 sub-iters of mfma+g2s to land it -> deep g2s pipeline CORRECT (vs default
                # 1-ahead which needs vmcnt~0). Requires caller prologue prefill buf2(k2)+soff=3*KSTEP.
                _ring2a = int(__import__("os").environ.get("FP4_WLRING_2A", "0"))
                _runr = 4 * int(__import__("os").environ.get("FP4_WLRING_U", "1"))  # unroll mult (amortize branch)
                for u in range(_runr):
                    cur_off = (u % 2) * set_sz
                    nxt_off = ((u + 1) % 2) * set_sz
                    rbuf = (u + 1) % 4
                    gbuf = ((u + 3) % 4) if _ring2a else ((u + 2) % 4)
                    # reg-read rbuf (refilled+published by PREV sub-iter end) into nxt set,
                    # interleaved into mfma(cur); g2s refill gbuf (read next sub-iter).
                    # FP4_WLNODSR probe: 1=skip ds_read, 2=skip g2s, 3=both (pure-mfma ceiling)
                    g2sl = [] if _NODSR in (2, 3) else (emit_g2s(gbuf, o_sa, o_sbl, o_sbr) + emit_scale_g2s(gbuf, 0))
                    _dsl_r = [] if _NODSR in (1, 3) else emit_ds(rbuf, nxt_off)
                    # FP4_WLRING_GBURST: hide ds_read (interleave) but BURST g2s after mfma (clean shadow
                    # for ds_read; g2s overlaps NEXT sub-iter via 2-ahead). probe: g2s-burst vs ds_read-burst.
                    # FP4_WLRING_GFRAC: aiter pattern -> g2s spread over FIRST gfrac of mfma (front-loaded,
                    # NOT end), ds_read over all, tail (1-gfrac) g2s-free so g2s lands before end-vmcnt.
                    _gfrac = float(__import__("os").environ.get("FP4_WLRING_GFRAC", "0"))
                    if _gfrac > 0:
                        _mm = emit_mm(cur_off); _n = len(_mm)
                        _greg = max(int(_n * _gfrac), len(g2sl)) if g2sl else 1
                        _gg = max(_greg // max(len(g2sl), 1), 1)
                        _dg = max(_n // max(len(_dsl_r), 1), 1)
                        _gi = _di = 0; _o = []
                        for _ix, _m in enumerate(_mm):
                            _o.append(_m)
                            if _di < len(_dsl_r) and _ix % _dg == 0:
                                _o.append(_dsl_r[_di]); _di += 1
                            if _gi < len(g2sl) and _ix < _greg and _ix % _gg == 0:
                                _o.append(g2sl[_gi]); _gi += 1
                        while _di < len(_dsl_r): _o.append(_dsl_r[_di]); _di += 1
                        while _gi < len(g2sl): _o.append(g2sl[_gi]); _gi += 1
                        L += _o
                    elif int(__import__("os").environ.get("FP4_WLRING_GBURST", "0")):
                        L += interleave(emit_mm(cur_off), _dsl_r) + g2sl
                    else:
                        L += interleave(emit_mm(cur_off), _dsl_r + g2sl)
                    # END: drain gbuf g2s + this reg-read (lgkmcnt0) + barrier. vmcnt(_WLV):
                    # _WLV=0 hard-correct; >0 soft (relies on g2s landing within the sub-iter,
                    # like the single-buffer's relaxed vmcnt) -> fewer stalls if det0 holds.
                    # FP4_WLRING_1BAR: barrier only on last sub-iter of unroll (1/trip vs 1/sub-iter)
                    _rbar = (u == _runr - 1) or (int(__import__("os").environ.get("FP4_WLRING_1BAR", "0")) == 0)
                    # FP4_WLRING_SPARSEVM: vmcnt only at unroll boundary (deep g2s pipeline like
                    # read-upfront/aiter) -> lgkmcnt(0) still per-sub-iter (ds_read ready). Needs
                    # buffering deep enough that g2s lands within the unroll (else racy).
                    _spvm = int(__import__("os").environ.get("FP4_WLRING_SPARSEVM", "0"))
                    if _spvm and u != _runr - 1:
                        _wc = "s_waitcnt lgkmcnt(0)"
                    else:
                        _wc = f"s_waitcnt vmcnt({_WLV}) lgkmcnt(0)"
                    L.append(_wc + ("\ns_barrier" if _rbar else ""))
                    for _so in (o_sa, o_sbl, o_sbr):
                        L.append(f"s_add_u32 ${_so}, ${_so}, ${i_kstep}")
                    for g in range(4):
                        L.append(f"s_add_u32 ${o_sca[g]}, ${o_sca[g]}, {2*_scstep if (_SCA2 and g==0) else _scstep}")
            elif _WLDB:
                # REGISTER DOUBLE-BUFFER (hide ds_read): mfma(set_cur) while ds_read(set_next)
                # interleaved -> read latency overlaps mfma. 1-ahead g2s->read RAW -> vmcnt(0).
                # vmcnt(_WLV): aiter keeps g2s deep-in-flight (vmcnt(10)); vmcnt(0) drains all
                # (correct only if g2s 1-ahead RAW). FP4_WLVMCN tunes pipeline depth (needs >=2-ahead g2s for correctness).
                _dbend = f"s_waitcnt vmcnt({_WLV}) lgkmcnt(0)\ns_barrier"
                # phase A: mfma set0 (k=2t), read buf1->set1 (k=2t+1), refill buf0 (k=2t+2)
                L += interleave(emit_mm(0), emit_ds(1, set_sz)
                                + emit_g2s(0, o_sa, o_sbl, o_sbr) + emit_scale_g2s(0, 0))
                L.append(_dbend)
                for g in range(4):
                    L.append(f"s_add_u32 ${o_sca[g]}, ${o_sca[g]}, {2*_scstep if (_SCA2 and g==0) else _scstep}")
                L.append(f"s_add_u32 ${o_ta}, ${o_sa}, ${i_kstep}")
                L.append(f"s_add_u32 ${o_tbl}, ${o_sbl}, ${i_kstep}")
                L.append(f"s_add_u32 ${o_tbr}, ${o_sbr}, ${i_kstep}")
                # phase B: mfma set1 (k=2t+1), read buf0->set0 (k=2t+2, just refilled), refill buf1
                L += interleave(emit_mm(set_sz), emit_ds(0, 0)
                                + emit_g2s(1, o_ta, o_tbl, o_tbr) + emit_scale_g2s(1, 0))
                L.append(_dbend)
                for g in range(4):
                    L.append(f"s_add_u32 ${o_sca[g]}, ${o_sca[g]}, {2*_scstep if (_SCA2 and g==0) else _scstep}")
                for _so in (o_sa, o_sbl, o_sbr):
                    L.append(f"s_add_u32 ${_so}, ${_so}, ${i_kstep}")
                    L.append(f"s_add_u32 ${_so}, ${_so}, ${i_kstep}")
            elif _INPLACE:
                # NEXT-K IN-PLACE REFILL (aiter mechanism, NSET=1, 2 LDS bufs, VGPR-feasible):
                # mfma(set0=k) while refilling set0 with k+1 from the OTHER buf after each A-temp's
                # last use (A-side ds_read overlaps mfma); B+scale drained at phase end. unroll-2.
                # buf0=even-k, buf1=odd-k (prologue fills buf0<-k0, buf1<-k1, like single-buffer).
                _elgk = int(__import__("os").environ.get("FP4_INPLACE_ELGK", "0"))  # leave late refills in flight
                # FP4_INPLACE_1BAR: phase A vmcnt-only (NO barrier), phase B vmcnt+barrier = 1
                # barrier/iter (vs 2). barriers cost ~20% of the INPLACE ceiling (pure-mfma probe).
                # safe: phase A g2s writes buf0 (read 2 phases later), barrier deferred to phase B.
                _1bar = int(__import__("os").environ.get("FP4_INPLACE_1BAR", "0"))
                _nobar = int(__import__("os").environ.get("FP4_NOBAR", "0"))  # ceiling probe: strip s_barrier
                _bar = "" if _nobar else "\ns_barrier"
                _ipend = f"s_waitcnt vmcnt({_WLV}) lgkmcnt({_elgk})" + _bar
                _ipenda = f"s_waitcnt vmcnt({_WLV}) lgkmcnt({_elgk})" + ("" if (_1bar or _nobar) else "\ns_barrier")
                _ipside = __import__("os").environ.get("FP4_INPLACE_ALT", "1") == "1"
                _diag = int(__import__("os").environ.get("FP4_INPLACE_DIAG", "0"))
                _sA = "DIAG" if _diag else "A"
                _sB = "DIAG" if _diag else ("B" if _ipside else "A")
                _ipnog = int(__import__("os").environ.get("FP4_WLNOG2S", "0"))  # probe: skip g2s (isolate ds_read cost)
                _noscg = int(__import__("os").environ.get("FP4_NOSCG2S", "0"))   # probe: skip ONLY scale g2s
                # TRB8: scale g2s (buffer_load->VGPR + vmcnt(0) + ds_write lane-major) must stay a
                # CONTIGUOUS block (the g2sl interleave would scramble its load->vmcnt->write order),
                # so it is appended after emit_inplace instead of folded into g2sl.
                _direct = _SCDWX4 and int(__import__("os").environ.get("FP4_SCDWX4_DIRECT", "0"))  # combined dwordx4-lds, no staging/ds_write
                def _scg(b): return [] if (_noscg or _TRB8 or _SCVGPR or (_SCDWX4 and not _direct)) else emit_scale_g2s(b, 0)
                _grup = int(__import__("os").environ.get("FP4_SCDWX4_GRUP", "1"))  # 1=GR upfront block (not interleaved); 0=GR in g2sl head
                # DIRECT: scale g2s interleaved in g2sl (end, like baseline). staging-SCDWX4: GR upfront/in-g2sl-head.
                _g2sA = [] if _ipnog else (((emit_scale_g2s(0,0) if (not _grup and not _direct) else []) + emit_g2s(0, o_sa, o_sbl, o_sbr) + (_scg(0) if _direct else [])) if _SCDWX4 else (emit_g2s(0, o_sa, o_sbl, o_sbr) + _scg(0)))
                _g2sB = [] if _ipnog else (((emit_scale_g2s(1,0) if (not _grup and not _direct) else []) + emit_g2s(1, o_ta, o_tbl, o_tbr) + (_scg(1) if _direct else [])) if _SCDWX4 else (emit_g2s(1, o_ta, o_tbl, o_tbr) + _scg(1)))
                # SCV2AHEAD: unroll-by-2 over par; 4-set double-buffer A0=0/A1=nsct/B0=2nsct/B1=3nsct.
                # par reads its set, loads the OTHER set (next same-phase kk, NOT in-flight) BEFORE the
                # mfma -> 2 vmcnt barriers before read (drains VMEM out-of-order) + no overwrite.
                _2a = _direct and int(__import__("os").environ.get("FP4_SCDWX4_2A", "0"))  # 4-buf 2-ahead deep prefetch (aiter pattern)
                _pars = [0, 1] if ((_SCVGPR and _SCV2AHEAD) or _2a) else [0]
                for _par in _pars:
                    if _2a:  # phase A: Q=2*par+0. refill reads buf[(Q+1)%4] (for next phase); g2s writes buf[(Q+2)%4]
                        _scrdbuf[0] = (2 * _par + 1) % 4
                        _g2sA = emit_g2s(0, o_sa, o_sbl, o_sbr) + emit_scale_g2s((2 * _par + 2) % 4, 0)
                    # phase A
                    if _SCVGPR:
                        if _SCV2AHEAD:
                            _scb[0] = _par * nsct                                  # read A[par]
                            L += emit_sc_vgpr((1 - _par) * nsct) + _scv_adv()      # load A[1-par] (diff buffer)
                        else:
                            _scb[0] = 0
                            _nop = int(__import__("os").environ.get("FP4_SCV_NOP","0"))
                            if _nop: L.append(f"s_nop {_nop}")
                            L += emit_sc_vgpr(nsct) + _scv_adv()                   # 1-ahead: load set B
                    if _SCDWX4 and _grup and not _direct:
                        L += emit_scale_g2s(0, 0)   # GR upfront (whole block before mfma, not interleaved)
                        if int(__import__("os").environ.get("FP4_SCDWX4_BAR2","0")): L.append("s_waitcnt vmcnt(0)" + ("" if int(__import__("os").environ.get("FP4_SCDWX4_DRONLY","0")) else "\ns_barrier"))
                    if _NSUBFOLD:
                        L += emit_nsubfold(0, 1, _g2sA)
                    else:
                        L += emit_inplace(1, _g2sA, side=_sA)
                    if _TRB8 and not _ipnog and not _noscg:
                        L += emit_scale_g2s(0, 0)   # contiguous scale g2s block for buffer 0
                    L.append(_ipenda)
                    if _SCDWX4:
                        if not _direct:
                            if int(__import__("os").environ.get("FP4_SCDWX4_LWBAR","1")) and int(__import__("os").environ.get("FP4_SCDWX4_BAR2","0")): L.append("s_waitcnt vmcnt(0) lgkmcnt(0)" + ("" if int(__import__("os").environ.get("FP4_SCDWX4_DRONLY","0")) else "\ns_barrier"))
                            L += emit_scale_lw(0)   # LW after _ipenda (GR landed by phase vmcnt, no vmcnt0)
                            L.append("s_waitcnt lgkmcnt(0)")  # drain LW before next-phase LR (else LR races LW)
                            if int(__import__("os").environ.get("FP4_SCDWX4_BAR","0")): L.append("s_barrier")
                        for g in (0, 2):  # preshuffled lane-contig kk stride (A=o_sca[0], B=o_sca[2])
                            L.append(f"s_add_u32 ${o_sca[g]}, ${o_sca[g]}, {_scvstep}")
                    elif _TRB8:  # lane_contig kk stride (A=o_sca[0], B=o_sca[2]); LDS via own ds_write
                        for g in (0, 2):
                            L.append(f"s_add_u32 ${o_sca[g]}, ${o_sca[g]}, {_scvstep}")
                    elif not _SCVGPR:
                        for g in range(4):
                            L.append(f"s_add_u32 ${o_sca[g]}, ${o_sca[g]}, {2*_scstep if (_SCA2 and g==0) else _scstep}")
                    L.append(f"s_add_u32 ${o_ta}, ${o_sa}, ${i_kstep}")
                    L.append(f"s_add_u32 ${o_tbl}, ${o_sbl}, ${i_kstep}")
                    L.append(f"s_add_u32 ${o_tbr}, ${o_sbr}, ${i_kstep}")
                    if _2a:  # phase B: Q=2*par+1. refill reads buf[(Q+1)%4]; g2s writes buf[(Q+2)%4]
                        _scrdbuf[0] = (2 * _par + 2) % 4
                        _g2sB = emit_g2s(1, o_ta, o_tbl, o_tbr) + emit_scale_g2s((2 * _par + 3) % 4, 0)
                    # phase B
                    if _SCVGPR:
                        if _SCV2AHEAD:
                            _scb[0] = 2 * nsct + _par * nsct                       # read B[par]
                            L += emit_sc_vgpr(2 * nsct + (1 - _par) * nsct) + _scv_adv()  # load B[1-par]
                        else:
                            _scb[0] = nsct
                            _nop = int(__import__("os").environ.get("FP4_SCV_NOP","0"))
                            if _nop: L.append(f"s_nop {_nop}")
                            L += emit_sc_vgpr(0) + _scv_adv()                      # 1-ahead: load set A
                    if _SCDWX4 and _grup and not _direct:
                        L += emit_scale_g2s(1, 0)   # GR upfront
                        if int(__import__("os").environ.get("FP4_SCDWX4_BAR2","0")): L.append("s_waitcnt vmcnt(0)" + ("" if int(__import__("os").environ.get("FP4_SCDWX4_DRONLY","0")) else "\ns_barrier"))
                    if _NSUBFOLD:
                        L += emit_nsubfold(1, 0, _g2sB)
                    else:
                        L += emit_inplace(0, _g2sB, side=_sB)
                    if _TRB8 and not _ipnog and not _noscg:
                        L += emit_scale_g2s(1, 0)   # contiguous scale g2s block for buffer 1
                    L.append(_ipend)
                    if _SCDWX4:
                        if not _direct:
                            if int(__import__("os").environ.get("FP4_SCDWX4_LWBAR","1")) and int(__import__("os").environ.get("FP4_SCDWX4_BAR2","0")): L.append("s_waitcnt vmcnt(0) lgkmcnt(0)" + ("" if int(__import__("os").environ.get("FP4_SCDWX4_DRONLY","0")) else "\ns_barrier"))
                            L += emit_scale_lw(1)   # ds_write buf1 from staging (GR landed by _ipend)
                            L.append("s_waitcnt lgkmcnt(0)")  # drain LW before next-phase LR
                            if int(__import__("os").environ.get("FP4_SCDWX4_BAR","0")): L.append("s_barrier")
                        for g in (0, 2):
                            L.append(f"s_add_u32 ${o_sca[g]}, ${o_sca[g]}, {_scvstep}")
                    elif _TRB8:  # lane_contig kk stride; LDS via own ds_write
                        for g in (0, 2):
                            L.append(f"s_add_u32 ${o_sca[g]}, ${o_sca[g]}, {_scvstep}")
                    elif not _SCVGPR:
                        for g in range(4):
                            L.append(f"s_add_u32 ${o_sca[g]}, ${o_sca[g]}, {2*_scstep if (_SCA2 and g==0) else _scstep}")
                    for _so in (o_sa, o_sbl, o_sbr):
                        L.append(f"s_add_u32 ${_so}, ${_so}, ${i_kstep}")
                        L.append(f"s_add_u32 ${_so}, ${_so}, ${i_kstep}")
            elif _BPREF:
                # ASYMMETRIC B-prefetch (hide B's ds_read on NATURAL emit_mm order → keep 6772 ceiling).
                # A+scale read-upfront(ready before mfma, exposed); B cross-iter ping-pong prefetch
                # (B[cur] ready from last phase, prefetch B[next] during this mfma → hidden). 48 operand.
                _bt = [(t_bl, t_br), (t_bl1, t_br1)]   # B temp bases per set
                def emit_a(buf):
                    return [f"ds_read_b128 ${t_a+ii*n_sub+s}, ${i_ab[buf][s]} offset:{ii*ts_a}"
                            for ii in range(nta) for s in range(n_sub)]
                def emit_sc(buf):
                    return [f"ds_read_b32 ${t_sc+slot}, ${i_scrb[buf]} offset:{slot*256}" for slot in range(nsct)]
                def emit_b(buf, bs):
                    tbl, tbr = _bt[bs]
                    r = [f"ds_read_b128 ${tbl+ji*n_sub+s}, ${i_blb[buf][s]} offset:{ji*ts_b}"
                         for ji in range(ntb) for s in range(n_sub)]
                    r += [f"ds_read_b128 ${tbr+ji*n_sub+s}, ${i_brb[buf][s]} offset:{ji*ts_b}"
                          for ji in range(ntb) for s in range(n_sub)]
                    return r
                def emit_mm_bp(bs):
                    tbl, tbr = _bt[bs]
                    r = []
                    for (sl, tb, sbfn) in ((0, tbl, sbl_t), (1, tbr, sbr_t)):
                        for s in range(n_sub):
                            for ii in range(nta):
                                for ji in range(ntb):
                                    q = sl * nq + ii * ntb + ji
                                    oa, ob = ii % 4, ji
                                    osel = f"op_sel:[{oa&1},{ob&1},0] op_sel_hi:[{(oa>>1)&1},{(ob>>1)&1},0]"
                                    r.append(f"v_mfma_scale_f32_16x16x128_f8f6f4 ${q}, ${t_a+ii*n_sub+s}, "
                                             f"${tb+ji*n_sub+s}, ${q}, ${sa_t(s,ii//4)}, ${sbfn(s)} {osel} cbsz:4 blgp:4")
                    return r
                _bend = f"s_waitcnt vmcnt({_WLV}) lgkmcnt(0)\ns_barrier"
                # FP4_BPREF_1BAR: phase-A barrier-less(vmcnt+lgkmcnt only) → 1 barrier/loop like
                # read-upfront, lets g2s overlap phaseA→B (2 barriers force g2s drain at phaseA).
                _1barbp = int(__import__("os").environ.get("FP4_BPREF_1BAR", "1"))
                _benda = (f"s_waitcnt vmcnt({_WLV}) lgkmcnt(0)" if _1barbp else _bend)
                _burst = int(__import__("os").environ.get("FP4_BPREF_BURST", "1"))
                _bpnod = int(__import__("os").environ.get("FP4_WLNODSR", "0"))   # probe: skip ds_read
                _bpnog = int(__import__("os").environ.get("FP4_WLNOG2S", "0"))   # probe: skip g2s
                _bphida = int(__import__("os").environ.get("FP4_BPREF_HIDEA", "0"))  # also burst-prefetch A? (needs 2nd A set - here just for probe)
                def _D(lst): return [] if _bpnod else lst
                def _G(buf, *a): return [] if _bpnog else (emit_g2s(buf, *a) + emit_scale_g2s(buf, 0))
                _arelax = int(__import__("os").environ.get("FP4_BPREF_ARELAX", "0"))  # speed-probe: relax A drain
                def _aread(buf):  # A+scale read (lgkmcnt-drained). probe: skippable
                    seg = _D(emit_a(buf)) + _D(emit_sc(buf))
                    return seg + ([f"s_waitcnt lgkmcnt({_arelax})"] if seg else [])
                # phase A: mfma(B set0=k2t), prefetch B set1<-buf1(k2t+1), g2s buf0<-k2t+2.
                # BURST: B-prefetch issued before mfma(reads ready B-set0) → mfma runs clean,
                # B-set1 ds_read overlaps in mfma EXECUTION shadow (not stealing issue slots).
                _pf0 = _D(emit_b(1, 1)); _pf1 = _D(emit_b(0, 0))
                if _burst:
                    L += _pf0 + interleave(emit_mm_bp(0), _G(0, o_sa, o_sbl, o_sbr))
                else:
                    L += interleave(emit_mm_bp(0), _pf0 + _G(0, o_sa, o_sbl, o_sbr))
                L.append(_benda)   # phase-A: barrier-less (1 barrier/loop, g2s overlaps into phase B)
                for g in range(4):
                    L.append(f"s_add_u32 ${o_sca[g]}, ${o_sca[g]}, {2*_scstep if (_SCA2 and g==0) else _scstep}")
                L.append(f"s_add_u32 ${o_ta}, ${o_sa}, ${i_kstep}")
                L.append(f"s_add_u32 ${o_tbl}, ${o_sbl}, ${i_kstep}")
                L.append(f"s_add_u32 ${o_tbr}, ${o_sbr}, ${i_kstep}")
                L += _aread(1)   # A+scale for phase B (exposed)
                # phase B: mfma(B set1=k2t+1), prefetch B set0<-buf0(k2t+2), g2s buf1<-k2t+3
                if _burst:
                    L += _pf1 + interleave(emit_mm_bp(1), _G(1, o_ta, o_tbl, o_tbr))
                else:
                    L += interleave(emit_mm_bp(1), _pf1 + _G(1, o_ta, o_tbl, o_tbr))
                L.append(_bend)
                for g in range(4):
                    L.append(f"s_add_u32 ${o_sca[g]}, ${o_sca[g]}, {2*_scstep if (_SCA2 and g==0) else _scstep}")
                for _so in (o_sa, o_sbl, o_sbr):
                    L.append(f"s_add_u32 ${_so}, ${_so}, ${i_kstep}")
                    L.append(f"s_add_u32 ${_so}, ${_so}, ${i_kstep}")
                L += _aread(0)   # A+scale for next iter phase A (exposed)
            else:
                # refill-SAME single-buffer: read buf, refill same buf 2-ahead. (NODSR/WLDSR probes)
                # phase A: read buf0, refill buf0 (k=2t+2, scale +0)
                if _NODSR:
                    L += interleave(emit_mm(), emit_g2s(0, o_sa, o_sbl, o_sbr) + emit_scale_g2s(0, 0))
                elif _WLDSR:
                    L += emit_phase_dsr(0, emit_g2s(0, o_sa, o_sbl, o_sbr) + emit_scale_g2s(0, 0))
                else:
                    L += emit_ds(0); L.append("s_waitcnt lgkmcnt(0)")
                    L += interleave(emit_mm(), emit_g2s(0, o_sa, o_sbl, o_sbr) + emit_scale_g2s(0, 0))
                L.append(_endpha)
                for g in range(4):
                    L.append(f"s_add_u32 ${o_sca[g]}, ${o_sca[g]}, {2*_scstep if (_SCA2 and g==0) else _scstep}")
                # phase B: read buf1, refill buf1 (k=2t+3, scale +_scstep via o_t*)
                L.append(f"s_add_u32 ${o_ta}, ${o_sa}, ${i_kstep}")
                L.append(f"s_add_u32 ${o_tbl}, ${o_sbl}, ${i_kstep}")
                L.append(f"s_add_u32 ${o_tbr}, ${o_sbr}, ${i_kstep}")
                if _NODSR:
                    L += interleave(emit_mm(), emit_g2s(1, o_ta, o_tbl, o_tbr) + emit_scale_g2s(1, 0))
                elif _WLDSR:
                    L += emit_phase_dsr(1, emit_g2s(1, o_ta, o_tbl, o_tbr) + emit_scale_g2s(1, 0))
                else:
                    L += emit_ds(1); L.append("s_waitcnt lgkmcnt(0)")
                    L += interleave(emit_mm(), emit_g2s(1, o_ta, o_tbl, o_tbr) + emit_scale_g2s(1, 0))
                L.append(_endph)
                for g in range(4):
                    L.append(f"s_add_u32 ${o_sca[g]}, ${o_sca[g]}, {2*_scstep if (_SCA2 and g==0) else _scstep}")
                for _so in (o_sa, o_sbl, o_sbr):
                    L.append(f"s_add_u32 ${_so}, ${_so}, ${i_kstep}")
                    L.append(f"s_add_u32 ${_so}, ${_so}, ${i_kstep}")
            if _ASYM:
                import math as _m2; _loopinc = (nbuf * nbuf_b) // _m2.gcd(nbuf, nbuf_b)
            elif ((_SCVGPR and _SCV2AHEAD) or (_SCDWX4 and int(__import__("os").environ.get("FP4_SCDWX4_2A", "0")))) and _INPLACE:
                _loopinc = 4   # INPLACE unroll-by-2 (par 0/1) -> body covers 4 BLOCK_K
            else:
                _loopinc = (nbuf if _SS else (4 * int(__import__("os").environ.get("FP4_WLRING_U", "1")) if _RING else 2))
            L.append(f"s_add_u32 ${o_cnt}, ${o_cnt}, {_loopinc}")
            L.append(f"s_cmp_lt_u32 ${o_cnt}, ${i_nval}")
            L.append("s_cbranch_scc1 1b")

            # FP4_PIN: pin operand+scale temps to contiguous physical VGPRs. Bypasses the LLVM RA
            # "Cannot decrease cascade number, illegal eviction" crash that NSET=2 (WLDB/RING)
            # triggers -> enables register double-buffer / ring on production (proven on bareasm).
            _PIN = int(__import__("os").environ.get("FP4_PIN", "0"))
            _PINBF = int(__import__("os").environ.get("FP4_PINBF", "0"))  # aiter layout: B low, A high
            _vtmp = ["=&v"] * ntmp2
            _nfrag_bp = na + 4 * nb   # BPREF: A(na) + B-set0(2nb) + B-set1(2nb)
            if _PIN and _BPREF:
                bv = int(__import__("os").environ.get("FP4_PINBASE", "8"))
                for j in range(_nfrag_bp):
                    _vtmp[j] = f"=&{{v[{bv}:{bv+3}]}}"; bv += 4
                for j in range(nsct):
                    _vtmp[_nfrag_bp + j] = f"=&{{v{bv}}}"; bv += 1
            elif _PIN and _SUB:
                bv = int(__import__("os").environ.get("FP4_PINBASE", "8"))
                _pbf = int(__import__("os").environ.get("FP4_PINBF", "0"))  # aiter bank: B frags LOW, A HIGH
                for ss in range(2):       # 2 reg sub-sets, each: nfs frags(vec4i32) + 4 scales(i32)
                    # frag layout: A=[0,nta), BL=[nta,nta+ntb), BR=[nta+ntb,nfs). PINBF -> B low, A high.
                    order = (list(range(nta, nfs)) + list(range(0, nta))) if _pbf else list(range(nfs))
                    for j in order:
                        _vtmp[ss * sub_sz + j] = f"=&{{v[{bv}:{bv+3}]}}"; bv += 4
                    for j in range(4):
                        _vtmp[ss * sub_sz + nfs + j] = f"=&{{v{bv}}}"; bv += 1
            elif _PIN:
                bv = int(__import__("os").environ.get("FP4_PINBASE", "8"))
                # frag order: A=[0,na), B(L+R)=[na,ntmp). PINBF -> B frags to LOW VGPR then A to
                # HIGH (mimic aiter A=v136+/B=v8-135 to test mfma read-port layout effect).
                # FP4_PINSC: pin scales to LOW VGPR first (gluon layout: scale v30/v32 low,
                # operands high), to test if scale-operand bank is the per-mfma stall source.
                _pinsc = int(__import__("os").environ.get("FP4_PINSC", "0"))
                for s in range(NSET):
                    order = (list(range(na, ntmp)) + list(range(0, na))) if _PINBF else list(range(ntmp))
                    if _pinsc:
                        # scales FIRST (low VGPR, contiguous). SCVGPR/TRB8 add a 2nd nsct block
                        # right after set0 so it lands at v[PINBASE+nsct:..] (emit_sc_vgpr needs
                        # the 2 scale sets contiguous at PINBASE / PINBASE+nsct).
                        _nsc2 = nsct * (4 if (_SCVGPR and _SCV2AHEAD) else (2 if (_TRB8 or _SCVGPR or _SCDWX4) else 1))
                        for j in range(_nsc2):
                            _vtmp[s * set_sz + ntmp + j] = f"=&{{v{bv}}}"; bv += 1
                        for j in order:
                            _vtmp[s * set_sz + j] = f"=&{{v[{bv}:{bv+3}]}}"; bv += 4
                    else:
                        for j in order:           # frags: vector<4xi32> = 4 VGPR
                            _vtmp[s * set_sz + j] = f"=&{{v[{bv}:{bv+3}]}}"; bv += 4
                        for j in range(nsct):           # scales: i32 = 1 VGPR
                            _vtmp[s * set_sz + ntmp + j] = f"=&{{v{bv}}}"; bv += 1
            cons = ",".join(
                ["=a"] * NT + _vtmp + ["=&s"] * 12  # accs, temps(ops+scale), cnt+3soff+3tmp+4scsoff+1sctmp
                + ["v"] * ((nbuf + 2 * nbuf_b) * n_sub)  # a(nbuf)/bl/br(nbuf_b) ds_read bases
                + ["s"] * (nbuf + 2 * nbuf_b)         # g2s dest bases a(nbuf)/bl/br(nbuf_b)
                + ["v"] * (nsa + nsb)                 # voffsets
                + ["s", "s", "s", "v", "s"]           # rsrc_a, rsrc_b, kstep, scv, nval
                + ["s", "s", "s"]                     # operand soffset inits A/BL/BR
                + ["v"] * _nscbuf                     # scale LDS read base × _nscbuf (4 for 2-ahead)
                + ["s"] * _nscbuf                     # scale LDS g2s dest base × _nscbuf
                + ["s", "s"]                          # scale rsrc A,B
                + ["v"]                               # scale voffset
                + ["s", "s", "s", "s"]                # scale soffset inits (A-g0,A-g1,BL,BR)
                + (["v"] * nbuf_b + ["v"] if _SCA2 else [])  # a2: A read base × nbuf_b + voffset
                + [str(q) for q in o_acc])            # tied accs
            if _BPREF:
                st = "!llvm.struct<(" + ", ".join(
                    ["vector<4xf32>"] * NT
                    + ["vector<4xi32>"] * (na + 4 * nb) + ["i32"] * nsct  # A + B-set0 + B-set1 + scale
                    + ["i32"] * 12) + ")>"
            elif _SUB:
                st = "!llvm.struct<(" + ", ".join(
                    ["vector<4xf32>"] * NT
                    + (["vector<4xi32>"] * nfs + ["i32"] * 4) * 2   # 2 sub-sets: nfs frags + 4 scales
                    + ["i32"] * 12) + ")>"
            else:
                st = "!llvm.struct<(" + ", ".join(
                    ["vector<4xf32>"] * NT
                    + (["vector<4xi32>"] * ntmp + ["i32"] * nsct
                       + ["i32"] * _scextra) * NSET  # +nsct TRB8/SCVGPR 2nd set; +3*nsct SCV2AHEAD 4-set double-buffer
                    + ["i32"] * 12) + ")>"
            _cache[key] = ("\n".join(L), cons, st)
            _df = __import__("os").environ.get("FP4_DUMPASM", "")
            if _df:
                with open(_df, "w") as _f:
                    _f.write("\n".join(L) + "\n\n;;;CONS:\n" + cons + "\n;;;ST:\n" + st)
        asm, cons, st = _cache[key]
        ins = []
        for b in range_constexpr(nbuf):           # A pool (nbuf)
            for s in range_constexpr(n_sub):
                ins.append(_raw(a_base[b][s]))
        for fr in (bl_base, br_base):             # B pool (nbuf_b)
            for b in range_constexpr(nbuf_b):
                for s in range_constexpr(n_sub):
                    ins.append(_raw(fr[b][s]))
        for b in range_constexpr(nbuf):           # g2s A dest (nbuf)
            ins.append(_raw(abase[b]))
        for fr in (blbase, brbase):               # g2s B dest (nbuf_b)
            for b in range_constexpr(nbuf_b):
                ins.append(_raw(fr[b]))
        for v in gl_a: ins.append(_raw(v))
        for v in gl_b: ins.append(_raw(v))
        ins.append(_raw(rsrc_a)); ins.append(_raw(rsrc_b))
        ins.append(_raw(kstep)); ins.append(_raw(scv)); ins.append(_raw(nval))
        ins.append(_raw(soff0)); ins.append(_raw(soff0_bl)); ins.append(_raw(soff0_br))
        for b in range_constexpr(_nscbuf): ins.append(_raw(sc_rb[b]))     # scale LDS read base (_nscbuf)
        for b in range_constexpr(_nscbuf): ins.append(_raw(sc_gb[b]))     # scale LDS g2s dest base
        ins.append(_raw(sc_rsa)); ins.append(_raw(sc_rsb))           # scale rsrc
        ins.append(_raw(sc_voff))                                     # scale voffset
        for g in range_constexpr(4): ins.append(_raw(sc_soff0[g]))    # scale soffset inits
        if _SCA2:
            for b in range_constexpr(nbuf_b): ins.append(_raw(sca_rb[b]))
            ins.append(_raw(sca_voff))
        for q in range_constexpr(nq): ins.append(_raw(cL[q]))
        for q in range_constexpr(nq): ins.append(_raw(cR[q]))
        r = _llvm.inline_asm(ir.Type.parse(st), ins, asm, cons, has_side_effects=True)
        o = [Vec(_llvm.extractvalue(ir.Type.parse("vector<4xf32>"), r, [q])) for q in range_constexpr(nq * 2)]
        return o[:nq], o[nq:]

    def call_mxfp4_wholeloop_8w(self, a0_base, a1_base, b0_base, b1_base, ts_a, ts_b,
                                ag0, ag1, bg0, bg1, gl_a, gl_b, rsrc_a, rsrc_b,
                                kstep, accs, n_sub, nsa, nsb, nval,
                                soff_a0, soff_a1, soff_b0, soff_b1,
                                sc_rb, sc_gb, sc_rsa, sc_rsb, sc_voff, sc_soff0, _cache={}):
        """8-WAVE WHOLE-LOOP bare-asm: entire K-loop = ONE inline-asm hw-loop, occ=2
        (waves_per_eu=2) hides ds_read latency via wave-switching so NO ring/double-buffer
        is needed (read-all-upfront + 2-buf ping-pong, unroll-2). Topology = 2x4 waves,
        4 quadrants/wave (a0/a1 M-regions x b0/b1 N-halves), nta=4 ntb=2 -> 8 accs/quad,
        32 accs total in VGPR (=&v; 8-wave can't hold AGPR for 32 accs under occ=2).
        Scale = 3 streams in LDS: A-r0, A-r1 (ScaleS2RPacked, 1 dword each, opsel=tile),
        B-comb (ScaleBCombPacked, ONE dword for both halves, opsel base 0/2). Real scale
        via ds_read_b32 (lgkmcnt, doesn't entangle g2s vmcnt). a*_base[b][s]/b*_base[b][s]:
        ds_read LDS addrs (b=buf0/1). ag*/bg*[b]: per-wave G2S LDS dest base SGPR. Returns
        the 32 updated accs (list)."""
        assert self.packed
        nta, ntb = self.n_tiles_a, self.n_tiles_b       # 4, 2
        nqq = nta * ntb                                  # 8 accs/quadrant
        NT = 4 * nqq                                     # 32 accs (4 quadrants)
        nar = nta * n_sub                                # A operand temps per region
        nbh = ntb * n_sub                                # B operand temps per half
        ntmp = 2 * nar + 2 * nbh                         # a0,a1,b0,b1 ds_read temps
        _NW = 8                                          # 8-wave: G2S step stride = n_waves*1024
        nbuf = len(a0_base)
        assert nbuf == 2
        _env = __import__("os").environ
        key = (nta, ntb, n_sub, nsa, nsb, ts_a, ts_b,
               _env.get("FP4_WLSYNC", "3"), _env.get("FP4_WLVMCN", "16"),
               _env.get("FP4_WLPRIO", "1"), _env.get("FP4_WLDSR", "0"),
               _env.get("FP4_WLDSRD", "8"), _env.get("FP4_WLNOG2S", "0"), _env.get("FP4_WLNODSR", "0"),
               _env.get("FP4_WLOTHER", "1"), _env.get("FP4_INPLACE", "0"),
               _env.get("FP4_INPLACE_ELGK", "0"), _env.get("FP4_INPLACE_SIDE", "DIAG"))
        if key not in _cache:
            o_acc = list(range(NT))
            t_a0 = NT; t_a1 = t_a0 + nar; t_b0 = t_a1 + nar; t_b1 = t_b0 + nbh
            nsct = 3 * n_sub                             # scale temps: A-r0, A-r1, B-comb x n_sub
            t_sc = t_b1 + nbh
            ntmp2 = ntmp + nsct
            o_cnt = NT + ntmp2
            o_sa0 = o_cnt + 1; o_sa1 = o_sa0 + 1; o_sb0 = o_sa1 + 1; o_sb1 = o_sb0 + 1
            o_ta0 = o_sb1 + 1; o_ta1 = o_ta0 + 1; o_tb0 = o_ta1 + 1; o_tb1 = o_tb0 + 1
            o_sca = [o_tb1 + 1 + g for g in range(3)]
            o_sct = o_sca[2] + 1
            nout = o_sct + 1
            # scale temp: group 0=A-r0, 1=A-r1, 2=B-comb; slot = grp*n_sub + s
            def sc_t(grp, s): return t_sc + grp * n_sub + s
            # inputs (after outputs):
            i = nout
            i_a0b = [[i + b * n_sub + s for s in range(n_sub)] for b in range(nbuf)]; i += nbuf * n_sub
            i_a1b = [[i + b * n_sub + s for s in range(n_sub)] for b in range(nbuf)]; i += nbuf * n_sub
            i_b0b = [[i + b * n_sub + s for s in range(n_sub)] for b in range(nbuf)]; i += nbuf * n_sub
            i_b1b = [[i + b * n_sub + s for s in range(n_sub)] for b in range(nbuf)]; i += nbuf * n_sub
            i_g_a0 = [i + b for b in range(nbuf)]; i += nbuf
            i_g_a1 = [i + b for b in range(nbuf)]; i += nbuf
            i_g_b0 = [i + b for b in range(nbuf)]; i += nbuf
            i_g_b1 = [i + b for b in range(nbuf)]; i += nbuf
            i_gla = [i + st for st in range(nsa)]; i += nsa
            i_glb = [i + st for st in range(nsb)]; i += nsb
            i_rsa = i; i += 1; i_rsb = i; i += 1
            i_kstep = i; i += 1; i_nval = i; i += 1
            i_sa0 = i; i += 1; i_sa1 = i; i += 1; i_sb0 = i; i += 1; i_sb1 = i; i += 1
            i_scrb = [i + b for b in range(nbuf)]; i += nbuf
            i_scgb = [i + b for b in range(nbuf)]; i += nbuf
            i_scrsa = i; i += 1; i_scrsb = i; i += 1
            i_scvoff = i; i += 1
            i_sca0 = [i + g for g in range(3)]; i += 3

            def emit_ds(buf):
                r = []
                for (tb, base) in ((t_a0, i_a0b), (t_a1, i_a1b)):
                    for ii in range(nta):
                        for s in range(n_sub):
                            r.append(f"ds_read_b128 ${tb + ii*n_sub+s}, ${base[buf][s]} offset:{ii*ts_a}")
                for (tb, base) in ((t_b0, i_b0b), (t_b1, i_b1b)):
                    for ji in range(ntb):
                        for s in range(n_sub):
                            r.append(f"ds_read_b128 ${tb + ji*n_sub+s}, ${base[buf][s]} offset:{ji*ts_b}")
                for slot in range(nsct):
                    r.append(f"ds_read_b32 ${t_sc + slot}, ${i_scrb[buf]} offset:{slot*256}")
                return r

            def emit_mm():
                r = []
                for ra, (a_tb, sa_grp) in enumerate(((t_a0, 0), (t_a1, 1))):
                    for rb, b_tb in enumerate((t_b0, t_b1)):
                        for s in range(n_sub):
                            for ii in range(nta):
                                for ji in range(ntb):
                                    q = (ra * 2 + rb) * nqq + ii * ntb + ji
                                    oa = ii            # A: opsel selects tile of the 4-tile packed dword
                                    ob = rb * 2 + ji   # B-comb: half rb (base 0/2) + tile ji
                                    osel = (f"op_sel:[{oa&1},{ob&1},0] "
                                            f"op_sel_hi:[{(oa>>1)&1},{(ob>>1)&1},0]")
                                    r.append(f"v_mfma_scale_f32_16x16x128_f8f6f4 ${q}, ${a_tb+ii*n_sub+s}, "
                                             f"${b_tb+ji*n_sub+s}, ${q}, ${sc_t(sa_grp,s)}, ${sc_t(2,s)} "
                                             f"{osel} cbsz:4 blgp:4")
                return r

            def emit_g2s(buf, oa0, oa1, ob0, ob1):
                r = []
                for (gbase, gl_list, rs, sop, nst) in (
                        (i_g_a0[buf], i_gla, i_rsa, oa0, nsa),
                        (i_g_a1[buf], i_gla, i_rsa, oa1, nsa),
                        (i_g_b0[buf], i_glb, i_rsb, ob0, nsb),
                        (i_g_b1[buf], i_glb, i_rsb, ob1, nsb)):
                    for st in range(nst):
                        r.append(f"s_add_u32 m0, ${gbase}, {st*_NW*1024}\n"
                                 f"buffer_load_dwordx4 ${gl_list[st]}, ${rs}, ${sop} offen lds")
                return r

            def emit_scale_g2s(buf, base_extra):
                # refill SC_lds[buf]: 3 groups (A-r0,A-r1,B-comb) x n_sub dwords.
                r = []
                for grp in range(3):
                    rsrc = i_scrsa if grp < 2 else i_scrsb
                    for s in range(n_sub):
                        slot = grp * n_sub + s
                        tot = base_extra + s * 256
                        if tot == 0:
                            r.append(f"s_add_u32 m0, ${i_scgb[buf]}, {slot*256}\n"
                                     f"buffer_load_dword ${i_scvoff}, ${rsrc}, ${o_sca[grp]} offen lds")
                        else:
                            r.append(f"s_add_u32 ${o_sct}, ${o_sca[grp]}, {tot}\n"
                                     f"s_add_u32 m0, ${i_scgb[buf]}, {slot*256}\n"
                                     f"buffer_load_dword ${i_scvoff}, ${rsrc}, ${o_sct} offen lds")
                return r

            def interleave(mm, g2s):
                gap = max(len(mm) // max(len(g2s), 1), 1)
                out = []; gi = 0
                for idx, m in enumerate(mm):
                    out.append(m)
                    if gi < len(g2s) and idx % gap == 0:
                        out.append(g2s[gi]); gi += 1
                while gi < len(g2s):
                    out.append(g2s[gi]); gi += 1
                return out

            def ds_line(buf, tt):
                if tt < t_a1:
                    rel = tt - t_a0; ii = rel // n_sub; s = rel % n_sub
                    return f"ds_read_b128 ${tt}, ${i_a0b[buf][s]} offset:{ii*ts_a}"
                if tt < t_b0:
                    rel = tt - t_a1; ii = rel // n_sub; s = rel % n_sub
                    return f"ds_read_b128 ${tt}, ${i_a1b[buf][s]} offset:{ii*ts_a}"
                if tt < t_b1:
                    rel = tt - t_b0; ji = rel // n_sub; s = rel % n_sub
                    return f"ds_read_b128 ${tt}, ${i_b0b[buf][s]} offset:{ji*ts_b}"
                if tt < t_sc:
                    rel = tt - t_b1; ji = rel // n_sub; s = rel % n_sub
                    return f"ds_read_b128 ${tt}, ${i_b1b[buf][s]} offset:{ji*ts_b}"
                slot = tt - t_sc
                return f"ds_read_b32 ${tt}, ${i_scrb[buf]} offset:{slot*256}"

            _WLD = int(__import__("os").environ.get("FP4_WLDSRD", "8"))   # ds_read-ahead depth

            def emit_phase_dsr(buf, g2sl):
                # interleave operand+scale ds_read INTO the mfma stream (staggered lgkmcnt)
                # so read latency overlaps the early mfmas; the WAR barrier is moved to
                # g2s_start (after all reads issued+landed) and refill-same g2s spreads
                # over the tail mfmas. mlist mirrors emit_mm with the consumed temps.
                mlist = []
                for ra, (a_tb, sa_grp) in enumerate(((t_a0, 0), (t_a1, 1))):
                    for rb, b_tb in enumerate((t_b0, t_b1)):
                        for s in range(n_sub):
                            for ii in range(nta):
                                for ji in range(ntb):
                                    q = (ra * 2 + rb) * nqq + ii * ntb + ji
                                    oa = ii; ob = rb * 2 + ji
                                    osel = (f"op_sel:[{oa&1},{ob&1},0] "
                                            f"op_sel_hi:[{(oa>>1)&1},{(ob>>1)&1},0]")
                                    at = a_tb + ii * n_sub + s; bt = b_tb + ji * n_sub + s
                                    sat = sc_t(sa_grp, s); sbt = sc_t(2, s)
                                    mline = (f"v_mfma_scale_f32_16x16x128_f8f6f4 ${q}, ${at}, ${bt}, "
                                             f"${q}, ${sat}, ${sbt} {osel} cbsz:4 blgp:4")
                                    mlist.append((mline, [at, bt, sat, sbt]))
                ro = []; pos = {}; first = {}
                for mi, (_ml, deps) in enumerate(mlist):
                    for tt in deps:
                        if tt not in pos:
                            pos[tt] = len(ro); ro.append(tt); first[tt] = mi
                iat = {}; prev = 0
                for tt in ro:
                    a = max(0, first[tt] - _WLD); a = max(a, prev); iat[tt] = a; prev = a
                sched = {}
                for tt in ro:
                    sched.setdefault(iat[tt], []).append(tt)
                g2s_start = (max(iat.values()) + 1) if iat else 0
                tail = max(len(mlist) - g2s_start, 1)
                gap = max(tail // max(len(g2sl), 1), 1)
                out = []; issued = 0; last = None; gi = 0; bar_done = False
                for mi, (ml, deps) in enumerate(mlist):
                    for tt in sched.get(mi, []):
                        out.append(ds_line(buf, tt)); issued += 1
                    need = issued - (max(pos[d] for d in deps) + 1)
                    if need < 0: need = 0
                    if need != last:
                        out.append(f"s_waitcnt lgkmcnt({need})"); last = need
                    if not bar_done and mi >= g2s_start:
                        out.append("s_waitcnt lgkmcnt(0)"); out.append("s_barrier")
                        bar_done = True; last = 0
                    out.append(ml)
                    if gi < len(g2sl) and mi >= g2s_start and (mi - g2s_start) % gap == 0:
                        out.append(g2sl[gi]); gi += 1
                if not bar_done:
                    out.append("s_waitcnt lgkmcnt(0)"); out.append("s_barrier")
                while gi < len(g2sl):
                    out.append(g2sl[gi]); gi += 1
                return out

            _INPLACE = int(__import__("os").environ.get("FP4_INPLACE", "0"))
            _ipnodsr8 = int(__import__("os").environ.get("FP4_WLNODSR", "0"))

            def emit_inplace_8w(nxt_buf, g2sl, side="DIAG"):
                # NEXT-K in-place refill ported to 8-wave (NSET=1, 2 LDS bufs). A-tiles=(ra,ii),
                # B-tiles=(rb,ji). After a tile's LAST mfma use, ds_read its NEXT-k tile from nxt_buf
                # into the freed reg -> overlap. side=DIAG: diagonal over A-idx x B-idx so both free
                # progressively; A/B: one side mid-refill, other end-drained. Accs order-independent.
                A_tiles = [(ra, ii) for ra in range(2) for ii in range(nta)]
                B_tiles = [(rb, ji) for rb in range(2) for ji in range(ntb)]
                nA, nB = len(A_tiles), len(B_tiles)
                def cinfo(ai, bi, s):
                    ra, ii = A_tiles[ai]; rb, ji = B_tiles[bi]
                    a_tb = t_a0 if ra == 0 else t_a1
                    b_tb = t_b0 if rb == 0 else t_b1
                    q = (ra * 2 + rb) * nqq + ii * ntb + ji
                    oa = ii; ob = rb * 2 + ji
                    osel = f"op_sel:[{oa&1},{ob&1},0] op_sel_hi:[{(oa>>1)&1},{(ob>>1)&1},0]"
                    at = a_tb + ii * n_sub + s; bt = b_tb + ji * n_sub + s
                    sat = sc_t(0 if ra == 0 else 1, s); sbt = sc_t(2, s)
                    ml = (f"v_mfma_scale_f32_16x16x128_f8f6f4 ${q}, ${at}, ${bt}, ${q}, "
                          f"${sat}, ${sbt} {osel} cbsz:4 blgp:4")
                    return ml, at, bt
                cells = []
                if side == "DIAG":
                    for s in range(n_sub):
                        for d in range(nA + nB - 1):
                            for ai in range(nA):
                                bi = d - ai
                                if 0 <= bi < nB:
                                    cells.append((ai, bi, s))
                else:
                    for s in range(n_sub):
                        outer = range(nA) if side == "A" else range(nB)
                        inner = range(nB) if side == "A" else range(nA)
                        for o_ in outer:
                            for i_ in inner:
                                ai, bi = (o_, i_) if side == "A" else (i_, o_)
                                cells.append((ai, bi, s))
                mlist = [cinfo(ai, bi, s) for (ai, bi, s) in cells]
                last = {}
                for mi, (_m, at, bt) in enumerate(mlist):
                    last[at] = mi; last[bt] = mi
                if side == "DIAG":
                    mid = set(t for t in last if t_a0 <= t < t_sc)
                elif side == "A":
                    mid = set(t for t in last if t_a0 <= t < t_b0)
                else:
                    mid = set(t for t in last if t_b0 <= t < t_sc)
                out = []; gi = 0; refilled = set()
                ngap = max(len(mlist) // max(len(g2sl), 1), 1)
                for mi, (ml, at, bt) in enumerate(mlist):
                    out.append(ml)
                    if not _ipnodsr8:
                        for rt in (at, bt):
                            if rt in mid and last[rt] == mi and rt not in refilled:
                                out.append(ds_line(nxt_buf, rt)); refilled.add(rt)
                    if gi < len(g2sl) and mi % ngap == 0:
                        out.append(g2sl[gi]); gi += 1
                while gi < len(g2sl):
                    out.append(g2sl[gi]); gi += 1
                if not _ipnodsr8:
                    for tt in range(t_a0, NT + ntmp2):
                        if tt not in refilled:
                            out.append(ds_line(nxt_buf, tt))
                return out

            _WLDSR = int(__import__("os").environ.get("FP4_WLDSR", "0"))
            _NOG2S = int(__import__("os").environ.get("FP4_WLNOG2S", "0"))  # ceiling probe: skip refill (garbage)
            def _g2s_all(buf, sa, sa1, sb, sb1):
                return [] if _NOG2S else (emit_g2s(buf, sa, sa1, sb, sb1) + emit_scale_g2s(buf, 0))
            _WLS = int(__import__("os").environ.get("FP4_WLSYNC", "3"))
            _WLV = int(__import__("os").environ.get("FP4_WLVMCN", "16"))
            if _WLS == 5:
                # 2 post-read barriers ONLY (no phase-end barrier). phase-A refill becomes
                # cross-wave visible at phase-B's post-read barrier (before next body's read);
                # vmcnt(0) here lands the refill. -> 2 barriers/body (vs 3-4).
                _endph = _endpha = f"s_waitcnt vmcnt({_WLV})"
            elif _WLS == 2:
                _endph = _endpha = "s_barrier"
            elif _WLS == 3:
                _endpha = f"s_waitcnt vmcnt({_WLV})"
                _endph = f"s_waitcnt vmcnt({_WLV})\ns_barrier"
            elif _WLS == 1:
                _endph = _endpha = f"s_waitcnt vmcnt({_WLV})\ns_barrier"
            else:
                _endph = _endpha = "s_waitcnt vmcnt(0)\ns_barrier"
            _WLOTHER = int(__import__("os").environ.get("FP4_WLOTHER", "0"))  # 0=refill-same (faster: 2-ahead relaxed vmcnt)
            if _WLOTHER:
                # refill-OTHER (read buf0/write buf1, like pipe cur/next): NO post-read
                # barrier (the ds_read-exposing stall); both phase-ends = vmcnt(0)+barrier
                # (RAW: sub1 reads buf1 written by sub0; WAR: sub1 writes buf0 read by sub0).
                _endph = _endpha = f"s_waitcnt vmcnt({_WLV})\ns_barrier"
            wA = 1 if _WLOTHER else 0   # phase-A WRITE buffer (other vs same)
            wB = 0 if _WLOTHER else 1
            _postbar = not _WLOTHER     # refill-same needs post-read WAR barrier

            L = [f"s_mov_b32 ${o_cnt}, 0",
                 f"s_mov_b32 ${o_sa0}, ${i_sa0}", f"s_mov_b32 ${o_sa1}, ${i_sa1}",
                 f"s_mov_b32 ${o_sb0}, ${i_sb0}", f"s_mov_b32 ${o_sb1}, ${i_sb1}"]
            for g in range(3):
                L.append(f"s_mov_b32 ${o_sca[g]}, ${i_sca0[g]}")
            _elgk8 = int(__import__("os").environ.get("FP4_INPLACE_ELGK", "0"))
            _sd8 = __import__("os").environ.get("FP4_INPLACE_SIDE", "DIAG")
            _ipend8 = f"s_waitcnt vmcnt({_WLV}) lgkmcnt({_elgk8})\ns_barrier"
            if _INPLACE:  # prologue: read buf0 (k=0) into operand regs (live) before the loop
                L += emit_ds(0); L.append("s_waitcnt lgkmcnt(0)\ns_barrier")
            L.append("1:")
            _scstep = n_sub * 256
            # phase A: read buf0, refill buf0 (k=2t+2)
            _SP = int(__import__("os").environ.get("FP4_WLPRIO", "1"))
            _NODSR = int(__import__("os").environ.get("FP4_WLNODSR", "0"))  # ceiling probe: skip ds_read (garbage)
            if _SP: L.append("s_setprio 1")
            if _INPLACE:
                # phase A: mfma k (regs), refill regs<-buf1 (k+1) in-place, g2s buf0<-k+2
                L += emit_inplace_8w(1, _g2s_all(0, o_sa0, o_sa1, o_sb0, o_sb1), side=_sd8)
            elif _NODSR:
                L += interleave(emit_mm(), _g2s_all(wA, o_sa0, o_sa1, o_sb0, o_sb1))
            elif _WLDSR:
                L += emit_phase_dsr(0, _g2s_all(wA, o_sa0, o_sa1, o_sb0, o_sb1))
            else:
                L += emit_ds(0); L.append("s_waitcnt lgkmcnt(0)")
                if _postbar:
                    L.append("s_barrier")  # refill-same WAR: all waves read buf0 before overwrite
                L += interleave(emit_mm(), _g2s_all(wA, o_sa0, o_sa1, o_sb0, o_sb1))
            if _SP: L.append("s_setprio 0")
            L.append(_ipend8 if _INPLACE else _endpha)
            for g in range(3):
                L.append(f"s_add_u32 ${o_sca[g]}, ${o_sca[g]}, {_scstep}")
            # phase B: read buf1, refill buf1 (k=2t+3 = +kstep)
            L.append(f"s_add_u32 ${o_ta0}, ${o_sa0}, ${i_kstep}")
            L.append(f"s_add_u32 ${o_ta1}, ${o_sa1}, ${i_kstep}")
            L.append(f"s_add_u32 ${o_tb0}, ${o_sb0}, ${i_kstep}")
            L.append(f"s_add_u32 ${o_tb1}, ${o_sb1}, ${i_kstep}")
            if _SP: L.append("s_setprio 1")
            if _INPLACE:
                # phase B: mfma k+1 (regs), refill regs<-buf0 (k+2) in-place, g2s buf1<-k+3
                L += emit_inplace_8w(0, _g2s_all(1, o_ta0, o_ta1, o_tb0, o_tb1), side=_sd8)
            elif _NODSR:
                L += interleave(emit_mm(), _g2s_all(wB, o_ta0, o_ta1, o_tb0, o_tb1))
            elif _WLDSR:
                L += emit_phase_dsr(1, _g2s_all(wB, o_ta0, o_ta1, o_tb0, o_tb1))
            else:
                L += emit_ds(1); L.append("s_waitcnt lgkmcnt(0)")
                if _postbar:
                    L.append("s_barrier")  # refill-same WAR: all waves read buf1 before overwrite
                L += interleave(emit_mm(), _g2s_all(wB, o_ta0, o_ta1, o_tb0, o_tb1))
            if _SP: L.append("s_setprio 0")
            L.append(_ipend8 if _INPLACE else _endph)
            for g in range(3):
                L.append(f"s_add_u32 ${o_sca[g]}, ${o_sca[g]}, {_scstep}")
            for _so in (o_sa0, o_sa1, o_sb0, o_sb1):
                L.append(f"s_add_u32 ${_so}, ${_so}, ${i_kstep}")
                L.append(f"s_add_u32 ${_so}, ${_so}, ${i_kstep}")
            L.append(f"s_add_u32 ${o_cnt}, ${o_cnt}, 2")
            L.append(f"s_cmp_lt_u32 ${o_cnt}, ${i_nval}")
            L.append("s_cbranch_scc1 1b")

            cons = ",".join(
                ["=&v"] * NT + ["=&v"] * ntmp2 + ["=&s"] * 13
                + ["v"] * (8 * n_sub)                 # a0,a1,b0,b1 ds_read bases x2 buf
                + ["s"] * 8                            # g2s dest bases (a0,a1,b0,b1) x2 buf
                + ["v"] * (nsa + nsb)                  # voffsets
                + ["s", "s", "s", "s"]                 # rsrc_a, rsrc_b, kstep, nval
                + ["s", "s", "s", "s"]                 # operand soffset inits a0,a1,b0,b1
                + ["v", "v"]                           # scale LDS read base (buf0,buf1)
                + ["s", "s"]                           # scale LDS g2s dest base (buf0,buf1)
                + ["s", "s"]                           # scale rsrc A,B
                + ["v"]                                # scale voffset
                + ["s", "s", "s"]                      # scale soffset inits (A-r0,A-r1,B-comb)
                + [str(q) for q in o_acc])
            st = "!llvm.struct<(" + ", ".join(
                ["vector<4xf32>"] * NT + ["vector<4xi32>"] * ntmp
                + ["i32"] * nsct + ["i32"] * 13) + ")>"
            _cache[key] = ("\n".join(L), cons, st)
        asm, cons, st = _cache[key]
        ins = []
        for fr in (a0_base, a1_base, b0_base, b1_base):
            for b in range_constexpr(2):
                for s in range_constexpr(n_sub):
                    ins.append(_raw(fr[b][s]))
        for fr in (ag0, ag1, bg0, bg1):
            for b in range_constexpr(2):
                ins.append(_raw(fr[b]))
        for v in gl_a: ins.append(_raw(v))
        for v in gl_b: ins.append(_raw(v))
        ins.append(_raw(rsrc_a)); ins.append(_raw(rsrc_b))
        ins.append(_raw(kstep)); ins.append(_raw(nval))
        ins.append(_raw(soff_a0)); ins.append(_raw(soff_a1))
        ins.append(_raw(soff_b0)); ins.append(_raw(soff_b1))
        for b in range_constexpr(2): ins.append(_raw(sc_rb[b]))
        for b in range_constexpr(2): ins.append(_raw(sc_gb[b]))
        ins.append(_raw(sc_rsa)); ins.append(_raw(sc_rsb))
        ins.append(_raw(sc_voff))
        for g in range_constexpr(3): ins.append(_raw(sc_soff0[g]))
        for q in range_constexpr(NT): ins.append(_raw(accs[q]))
        r = _llvm.inline_asm(ir.Type.parse(st), ins, asm, cons, has_side_effects=True)
        return [Vec(_llvm.extractvalue(ir.Type.parse("vector<4xf32>"), r, [q])) for q in range_constexpr(NT)]

    def call_iter_asm(self, a0, a1, b0, b1, sa0, sa1, sb0, sb1, accs, n_sub, _cache={}):
        """MONOLITHIC per-iter asm cluster (bareasm campaign): ALL 4 quadrants' 64 MFMAs
        (nta*ntb*n_sub each) in ONE inline-asm block, 32 accs in AGPR (=a) tied across
        iters (caller threads accs iter->iter; extract only once/iter -> stays AGPR, no
        round-trip -> the RA disaster of per-quadrant =a is avoided; see memory _asm6b).
        Frees 128 VGPR for B double-buffer / G2S co-issue (added later). a0/a1[i][s],
        b0/b1[j][s] i32x4 (pad=False); sa0/sa1[s] packed-i32; sb0/sb1[s]=(packed, ob)."""
        assert self.packed
        nta, ntb = self.n_tiles_a, self.n_tiles_b
        nq = nta * ntb           # accs per quadrant
        NT = 4 * nq              # 32 total
        na = nta * n_sub         # a frags per region
        nb = ntb * n_sub         # b frags per region
        ob0, ob1 = sb0[0][1], sb1[0][1]
        _agpr = __import__("os").environ.get("FP4_AGPR", "1") == "1"
        key = (nta, ntb, n_sub, ob0, ob1, _agpr)
        if key not in _cache:
            b_a0, b_a1 = NT, NT + na
            b_b0, b_b1 = NT + 2 * na, NT + 2 * na + nb
            b_sa0 = NT + 2 * na + 2 * nb
            b_sa1, b_sb0, b_sb1 = b_sa0 + n_sub, b_sa0 + 2 * n_sub, b_sa0 + 3 * n_sub
            quads = [  # (acc_base, a_base, b_base, sa_base, sb_base, ob)
                (0 * nq, b_a0, b_b0, b_sa0, b_sb0, ob0),
                (1 * nq, b_a0, b_b1, b_sa0, b_sb1, ob1),
                (2 * nq, b_a1, b_b0, b_sa1, b_sb0, ob0),
                (3 * nq, b_a1, b_b1, b_sa1, b_sb1, ob1),
            ]
            L = []
            for (ab, a_b, b_b, sa_b, sb_b, ob) in quads:
                for s in range(n_sub):
                    for i in range(nta):
                        for j in range(ntb):
                            q = ab + i * ntb + j
                            oa, obj = i, ob + j
                            osel = (f"op_sel:[{oa & 1},{obj & 1},0] "
                                    f"op_sel_hi:[{(oa >> 1) & 1},{(obj >> 1) & 1},0]")
                            L.append(f"v_mfma_scale_f32_16x16x128_f8f6f4 ${q}, ${a_b + i * n_sub + s}, "
                                     f"${b_b + j * n_sub + s}, ${q}, ${sa_b + s}, ${sb_b + s} {osel} cbsz:4 blgp:4")
            _ac = "=a" if _agpr else "=v"
            cons = ",".join([_ac] * NT + ["v"] * (2 * na + 2 * nb + 4 * n_sub) + [str(q) for q in range(NT)])
            st = "!llvm.struct<(" + ", ".join(["vector<4xf32>"] * NT) + ")>"
            _cache[key] = ("\n".join(L), cons, st)
        asm, cons, st = _cache[key]
        ins = []
        for fr in (a0, a1):
            for i in range_constexpr(nta):
                for s in range_constexpr(n_sub):
                    ins.append(_raw(fr[i][s]))
        for fr in (b0, b1):
            for j in range_constexpr(ntb):
                for s in range_constexpr(n_sub):
                    ins.append(_raw(fr[j][s]))
        for sc in (sa0, sa1):
            for s in range_constexpr(n_sub):
                ins.append(_raw(sc[s]))
        for sc in (sb0, sb1):
            for s in range_constexpr(n_sub):
                ins.append(_raw(sc[s][0]))
        for q in range_constexpr(NT):
            ins.append(_raw(accs[q]))
        r = _llvm.inline_asm(ir.Type.parse(st), ins, asm, cons, has_side_effects=True)
        return [Vec(_llvm.extractvalue(ir.Type.parse("vector<4xf32>"), r, [q])) for q in range_constexpr(NT)]

    def call_one(self, a, b, c, sa, sb, i, j, s):
        """Single (i,j) sub-block-s MFMA for fine-grained G2S-coissue interleaving
        (4wave-style: hand-place one MFMA between G2S/S2R load_one calls). a[i][s]/
        b[j][s] nested frags; sa/sb same shape as call_subs. Returns the new c_ij
        accumulator (caller does c[idx(i,j)] = call_one(...))."""
        cij = c[self.idx(i, j)]
        if self.packed:
            sb_p, ob = sb[s]
            return self._do_packed(a[i][s], b[j][s], cij, sa[s], i, sb_p, ob + j)
        return self._do(a[i][s], b[j][s], cij, sa[s][i], sb[s][j])


def fp4_g2s_contiguous_offsets(lane_id, wave_id, n_steps):
    """Per-lane gmem byte offsets for a WEIGHT-PRESHUFFLED B G2S: the host laid B out
    in exactly LDS-fill order (see preshuffle_b_mxfp4), so each lane reads a contiguous
    16B at wave*1024 + step*(n_waves*1024) + lane*16 -> the 64-lane G2S step is ONE
    coalesced 1024B read (vs 8 scattered K2-strided cache lines in the identity path).
    Kills the B-side G2S vmcnt stall that caps the in-LDS pipe ~4910."""
    n_waves = fx.block_dim.x // 64
    return [wave_id * 1024 + r * (n_waves * 1024) + lane_id * 16 for r in range_constexpr(n_steps)]


def preshuffle_b_mxfp4(B, N, K, BLOCK_N=256, BLOCK_K=256, swizzle=True):
    """Host: reorder B[N, K//2] (packed fp4 uint8) into the kernel's contiguous G2S
    order so the device B G2S reads coalesced (and the resulting LDS matches the
    identity/swizzled layout the S2R expects -- the swizzle sigma is baked in here).
    Mirrors fp4_g2s_offsets(swizzle=) gather. Returns a flat uint8 tensor."""
    import torch
    K2 = K // 2
    BPR = BLOCK_K // 2
    KSTEP = BPR
    LDS_BLOCK_N = BLOCK_N // 2
    n_waves = 8
    lpr = BPR // 16
    rows_per_step = 64 // lpr
    N_LDS_STEPS_B = LDS_BLOCK_N // (rows_per_step * n_waves)
    Kit = K // BLOCK_K
    nbn = N // BLOCK_N
    dev = B.device
    B16 = B.view(N, K2 // 16, 16)
    lane = torch.arange(64, device=dev)
    if swizzle:
        ph = lane // 8
        cib = ph * 8 + (lane % 8 - ph + 8) % 8  # _swz_inv on lane
    else:
        cib = lane
    bn = torch.arange(nbn, device=dev).view(nbn, 1, 1, 1, 1, 1)
    region = torch.arange(2, device=dev).view(1, 2, 1, 1, 1, 1)
    kk = torch.arange(Kit, device=dev).view(1, 1, Kit, 1, 1, 1)
    rr = torch.arange(N_LDS_STEPS_B, device=dev).view(1, 1, 1, N_LDS_STEPS_B, 1, 1)
    wv = torch.arange(n_waves, device=dev).view(1, 1, 1, 1, n_waves, 1)
    ln = cib.view(1, 1, 1, 1, 1, 64)
    row_in_region = ln // lpr + wv * rows_per_step + rr * (rows_per_step * n_waves)
    Nrow = (bn * BLOCK_N + region * LDS_BLOCK_N + row_in_region).expand(nbn, 2, Kit, N_LDS_STEPS_B, n_waves, 64)
    Kblk = (kk * (KSTEP // 16) + (ln % lpr)).expand(nbn, 2, Kit, N_LDS_STEPS_B, n_waves, 64)
    gathered = B16[Nrow.reshape(-1), Kblk.reshape(-1)]  # [total, 16]
    return gathered.reshape(-1).contiguous()


def preshuffle_b_vgpr(B, N, K, BLOCK_N=256, BLOCK_K=256):
    """Host: reorder B[N,K//2] into B->VGPR-direct consumption order so each lane can
    buffer_load its exact MFMA b-operands (i32x4) coalesced, bypassing B's LDS round-trip
    (no buffer_load_lds, no b ds_read). Order = (block_n, region, k, wave_n, lane, tile,
    sub, 16B), matching BVgprLoader / b_s2r(wave_n). Returns flat uint8."""
    import torch
    K2 = K // 2
    KSTEP = BLOCK_K // 2
    LDS_BLOCK_N = BLOCK_N // 2
    N_TILES_B = BLOCK_N // 128
    N_SUB = BLOCK_K // 128
    KI = K // BLOCK_K
    nbn = N // BLOCK_N
    NWN = 4  # N-waves (2x4 topo)
    dev = B.device
    B16 = B.view(N, K2 // 16, 16)
    bn = torch.arange(nbn, device=dev).view(nbn, 1, 1, 1, 1, 1, 1)
    reg = torch.arange(2, device=dev).view(1, 2, 1, 1, 1, 1, 1)
    kk = torch.arange(KI, device=dev).view(1, 1, KI, 1, 1, 1, 1)
    wn = torch.arange(NWN, device=dev).view(1, 1, 1, NWN, 1, 1, 1)
    ln = torch.arange(64, device=dev).view(1, 1, 1, 1, 64, 1, 1)
    ti = torch.arange(N_TILES_B, device=dev).view(1, 1, 1, 1, 1, N_TILES_B, 1)
    su = torch.arange(N_SUB, device=dev).view(1, 1, 1, 1, 1, 1, N_SUB)
    lane16 = ln % 16
    g = ln // 16
    Nrow = bn * BLOCK_N + reg * LDS_BLOCK_N + wn * (N_TILES_B * 16) + ti * 16 + lane16
    Kblk = kk * (KSTEP // 16) + su * 4 + g  # (k*KSTEP + s*64 + g*16)//16
    shp = (nbn, 2, KI, NWN, 64, N_TILES_B, N_SUB)
    gathered = B16[Nrow.expand(shp).reshape(-1), Kblk.expand(shp).reshape(-1)]
    return gathered.reshape(-1).contiguous()


class BVgprLoader:
    """B->VGPR-direct loader: buffer_load lane's MFMA b-operands straight from the
    host-preshuffled B (preshuffle_b_vgpr), no LDS. .load(region, k) returns frag[tile][sub]
    (i32x8 padded), drop-in for b_s2r.load(b_cur{region})."""

    def __init__(self, B_tensor, block_n_var, KI, n_tiles_b, n_sub, num_bytes, pad=True):
        self.lane = fx.thread_idx.x % 64
        self.wave_n = (fx.thread_idx.x // 64) % 4
        self.block_n = block_n_var
        self.KI = KI
        self.n_tiles_b = n_tiles_b
        self.n_sub = n_sub
        self.pad = pad  # True -> i32x8 (intrinsic MFMA); False -> i32x4 (asm cbsz:4)
        self.fpl = n_tiles_b * n_sub  # frags per lane
        self.rsrc = buffer_ops.create_buffer_resource(B_tensor, max_size=False, num_records_bytes=num_bytes)
        self.zero4 = Vec.filled(4, 0, fx.Int32)

    def load(self, region, k):
        # i32 index: (((((bn*2+region)*KI+k)*4+wave_n)*64+lane)*fpl + (i*n_sub+s)) * 4
        base = (((self.block_n * 2 + region) * self.KI + k) * 4 + self.wave_n) * 64 + self.lane
        base = base * self.fpl
        frag = []
        for i in range_constexpr(self.n_tiles_b):
            subs = []
            for s in range_constexpr(self.n_sub):
                idx = (base + (i * self.n_sub + s)) * 4  # *4: 16B = 4 i32
                v4 = Vec(buffer_ops.buffer_load(self.rsrc, idx, vec_width=4, dtype=T.i32))
                subs.append(pack_i32x4_i32x8(v4, self.zero4) if const_expr(self.pad) else v4)
            frag.append(subs)
        return frag


def _swz_fwd(c):
    """LDS bank-swizzle, logical chunk -> physical slot. Modular-add rotate within
    each 1024B block: phase = c//8 (= row//2, the 128B-line index within the block,
    max_phase 8), rotate the low 3 bits (the 8-chunk = 128B bank period) by phase.
    The 8 same-bank rows of one ds_read_b128 (which collide in identity layout) then
    land on all 8 bank-groups. Bijection; involution-free, so G2S uses _swz_inv.
    Direct-DMA safe: it permutes the lane<->gmem assignment, footprint/stride/coalesce
    unchanged.
    n_sub-agnostic (applied on the cib = byte-offset chunk index, so it covers BK128's
    8-way AND BK256's 16-way (128B row-stride == bank period) ds_read conflict)."""
    ph = c // 8
    return ph * 8 + (c % 8 + ph) % 8


def _swz_inv(c):
    """Inverse of _swz_fwd (physical slot -> logical chunk). G2S lane L writes the
    contiguous physical slot L, so it must fetch the gmem of logical chunk _swz_inv(L)
    for S2R's _swz_fwd read to land on it."""
    ph = c // 8
    return ph * 8 + (c % 8 + 8 - ph) % 8


def fp4_g2s_offsets(lane_id, wave_id, K, n_steps, bytes_per_row, swizzle=False):
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
        cib = _swz_inv(lane_id) if swizzle else lane_id  # physical slot -> logical chunk
        row = cib // lpr + wave_id * rows_per_step + r * (n_waves * rows_per_step)
        chunk = cib % lpr
        offs.append(row * (K // 2) + chunk * 16)
    return offs


def recommend_config(M, N, K):
    """Data-driven production config for dense mxfp4 (mode=pipe) on MI355X.

    Returns (BLOCK_M, BLOCK_N, group_m, group_n, num_xcds, block_k, swizzle).
    Verified-solid levers:

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
    # (nb=16) nb//8=2 -> gn=4 confirmed +1.0%/+1.5% (M4096/M8192, 15/15 both).
    # round-77 (same GOLD sweep): 70B down (nb=32, K=28672) wants the ALIGNED
    # (num_xcds=16, group_n=2) = 16 narrow bands, NOT (nx8,gn4): GOLD +1.2%/+0.8%
    # (15/15 both M). This is a genuine JOINT-nx lever — the mis-aligned (nx8,gn2)
    # LOSES -5.8%, so gn=2 is ONLY a win paired with nx=16 (#bands==num_xcds keeps
    # each XCD's pid-block on one N-band). big-N (nb>=96) keeps nb//8=14 / nx8.
    # num_xcds default 8 = physical XCD count (r38/r56); only 70B down overrides to 16.
    num_xcds = 8
    if nb >= 96:
        group_n = nb // 8
    elif K >= 28672:
        group_n = 2
        num_xcds = 16
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
    # BK256 + LDS bank-swizzle (BN256 bulk only): BK256 (n_sub=2) amortizes the per-iter
    # G2S-setup MFMA-bubble (G2S cost 28%->11%), but its 128B LDS row-stride == bank
    # period -> 16-way ds_read bank conflict (57% GPUTime stall); the swizzle kills it
    # (3.0->0.0 conf/access). Neither alone helps bulk; COMBINED = real-scale +7.5~14.5%
    # across all BN256 shapes (8192^2 K28672 4280->4905, bit-exact, no-G2S ceiling 6165).
    # BN128 (narrow kv) stays BK128/no-swizzle (swizzle asserts BN256; BK256 doubles LDS
    # and the narrow grid is occupancy- not bubble-bound). See feedback memory 2026-06-11.
    block_k = 256 if block_n == 256 else 128
    swizzle = block_n == 256
    return block_m, block_n, group_m, group_n, num_xcds, block_k, swizzle


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


class S2RLoaderFp4:
    """LDS->reg fp4 fragment loader (identity LDS, bytes_per_row K-iter rows).

    A K-iter spans n_sub == BLOCK_K/128 128-K sub-blocks; each sub-block s is one
    16x16x128 MFMA. Lane (g=lane//16, r=lane%16) reads, per sub-block, its block
    g == 16 contiguous packed bytes (32 fp4) at row*bpr + s*64 + g*16, bitcast to
    i32x4 then padded to i32x8 (low 16B real, upper zero) for the cbsz=4 operand.
    ``load`` returns frag[tile][sub] (nested).
    """

    def __init__(self, wave_idx, n_tiles, n_sub, bytes_per_row, row_stride=None, pad=True, swizzle=False):
        self.lane16 = fx.thread_idx.x % 16
        self.g = (fx.thread_idx.x % 64) // 16
        self.wave_idx = wave_idx
        self.n_tiles = n_tiles
        self.n_sub = n_sub
        self.bpr = bytes_per_row
        self.row_stride = row_stride if row_stride is not None else bytes_per_row
        self.pad = pad  # False -> return raw i32x4 (for inline-asm MFMA; fp4 HW needs only 16B)
        self.swizzle = swizzle  # bank-swizzle read offset (pairs with G2S _swz_inv); BK128+BK256
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
                if const_expr(self.swizzle):
                    # General byte-offset swizzle (n_sub-agnostic): permute 16B chunks
                    # within each 1024B (8-bank-period) block. Works for BK128
                    # (row_stride=64 -> 16 rows x 4 chunks) AND BK256 (row_stride=128 ->
                    # 8 rows x 8 chunks, where 128B stride == bank period = 16-way
                    # conflict that this kills). cib//8 = row-within-block = phase.
                    off_nat = row * self.row_stride + s * 64 + self.g * 16
                    cib = (off_nat % 1024) // 16
                    off = (off_nat // 1024) * 1024 + _swz_fwd(cib) * 16
                else:
                    off = row * self.row_stride + s * 64 + self.g * 16
                v4 = self._load16(lds_src, off).bitcast(fx.Int32)
                subs.append(v4 if not self.pad else pack_i32x4_i32x8(v4, self.zero4))
            frag.append(subs)
        return frag

    def load_one(self, lds_src, i, s):
        """Single tile-i sub-block-s ds_read (mirrors load()'s per-(i,s) body incl.
        swizzle) for 4wave-style fine-grained G2S-coissue interleaving. Returns the
        i32x4 (pad=False) or i32x8 (pad=True) frag for one MFMA operand."""
        row = self.wave_idx * (self.n_tiles * 16) + i * 16 + self.lane16
        if const_expr(self.swizzle):
            off_nat = row * self.row_stride + s * 64 + self.g * 16
            cib = (off_nat % 1024) // 16
            off = (off_nat // 1024) * 1024 + _swz_fwd(cib) * 16
        else:
            off = row * self.row_stride + s * 64 + self.g * 16
        v4 = self._load16(lds_src, off).bitcast(fx.Int32)
        return v4 if not self.pad else pack_i32x4_i32x8(v4, self.zero4)

    def addr(self, lds_src, s=0):
        """LDS byte-address (i32 ptrtoint) of each tile's sub-block-`s` fragment, for
        feeding ds_read into an inline-asm MFMA cluster. Mirrors load()'s offset."""
        out = []
        for i in range_constexpr(self.n_tiles):
            row = self.wave_idx * (self.n_tiles * 16) + i * 16 + self.lane16
            off_nat = row * self.row_stride + s * 64 + self.g * 16
            if const_expr(self.swizzle):   # match load()'s swizzle so asm ds_read reads right
                cib = (off_nat % 1024) // 16
                off = (off_nat // 1024) * 1024 + _swz_fwd(cib) * 16
            else:
                off = off_nat
            i8_iter = fx.recast_iter(fx.Uint8, fx.add_offset(lds_src.ptr, fx.make_int_tuple(off)))
            out.append(fx.ptrtoint(i8_iter))
        return out

    def base_addr(self, lds_src, s=0):
        """Single base LDS address (tile 0, sub-block s). Per-tile fragments are at
        base + i*tile_stride (tile_stride = 16*row_stride bytes) -> the asm uses ONE
        address reg per region + a ds_read offset immediate, instead of n_tiles full
        address regs (cuts the address-register VGPR pressure that caused spill)."""
        row0 = self.wave_idx * (self.n_tiles * 16) + self.lane16
        off_nat = row0 * self.row_stride + s * 64 + self.g * 16
        if const_expr(self.swizzle):   # tile i = base + i*tile_stride stays swz-correct
            cib = (off_nat % 1024) // 16   # (tile_stride is a 1024-multiple -> %1024 const)
            off = (off_nat // 1024) * 1024 + _swz_fwd(cib) * 16
        else:
            off = off_nat
        i8_iter = fx.recast_iter(fx.Uint8, fx.add_offset(lds_src.ptr, fx.make_int_tuple(off)))
        return fx.ptrtoint(i8_iter)

    @property
    def tile_stride(self):
        return 16 * self.row_stride


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
    mode: str = "pipe",
    block_k: int = 128,
    group_m: int = 4,
    num_xcds: int = 8,
    group_n: int = 0,
    swizzle: bool = False,
    interleave: bool = False,
    preshuffle_b: bool = False,
    b_vgpr: bool = False,
    asm_mma: bool = False,
    asm_iter: bool = False,
):
    # block_k: logical fp4 contracted per K-iter. The 16x16x128 MFMA always does
    # 128 K, so a K-iter spans N_SUB == block_k/128 sub-block MFMAs per accumulator.
    BLOCK_K = block_k
    # FP4_STAGE3: gluon-style 3-stage / 1-barrier-per-iter K-loop with 3-deep LDS ring.
    # k+2 G2S targets a 3rd buffer (free, last read at k-1) -> removes the WAR that forced
    # the 2-stage's 7 s_barriers; keep 1 cross-wave wait_barrier/iter + sched cadence ->
    # LLIR scheduler buries ds_read in the MFMA shadow (the gluon 4274->5253 lever).
    # Needs BLOCK_K==128 (3-deep BK256=192KB>160KB LDS; BK128=96KB fits).
    stage3 = int(__import__("os").environ.get("FP4_STAGE3", "0")) == 1
    if stage3:
        assert mode == "pipe" and BLOCK_M == 256 and BLOCK_N == 256 \
            and not (asm_mma or asm_iter or interleave or b_vgpr or swizzle or preshuffle_b), \
            "FP4_STAGE3 needs pipe/BM256/BN256, no asm/interleave/bvgpr/swizzle/preshuffle_b"
    # B6: BLOCK_M=192 (small-M occupancy for grid-underfilled kv) needs only
    # BLOCK_M % 64 == 0 (N_TILES_A = BLOCK_M//64 integral) + (BLOCK_M//2) % 16 == 0
    # (LDS region a 16-row-wave granular). BM192 -> N_TILES_A=3, LDS_BLOCK_M=96.
    assert BLOCK_M >= 128 and BLOCK_M % 64 == 0 and (BLOCK_M // 2) % 16 == 0
    assert BLOCK_N >= 128 and BLOCK_N % 128 == 0
    assert BLOCK_K % 128 == 0 and K % BLOCK_K == 0

    K_ITERS = K // BLOCK_K
    KI = K_ITERS
    # Packed E8M0 + opsel (4x less scale VMEM) is the default; auto-disabled for
    # configs that can't express it (kv BN128 -> broadcast-dwordx4 path). Host must
    # match via _packed_eligible() / preshuffle_mxfp4_scales().
    packed_scale = _packed_eligible(mode, BLOCK_N)
    # combine-A: both M-regions' packed scale in one coalesced dwordx2 (2 loads/iter
    # instead of 3). DISABLED: the A2 layout (ScaleS2RPackedA2/preshuffle_scale_packed_a2)
    # is 4-wave geometry (consecutive groups, wave stride 128); on 8-wave (wave stride 64,
    # region +2 groups) wave_m collapses -> 3/4 (wave_m,region) load wrong A-scale -> SNR~1.8.
    # Per-region ScaleS2RPacked is correct for 8-wave (= 4-wave production path).
    combine_a = False
    N_SUB = BLOCK_K // 128  # 128-K MFMA sub-blocks per K-iter
    BPR = BLOCK_K // 2  # packed-fp4 bytes per K-iter row in LDS
    KSTEP = BPR  # gmem byte stride per K-iter
    # LDS bank-swizzle kills the identity-layout ds_read bank conflict via the
    # direct-DMA lane<->gmem permutation (see _swz_fwd); BM256/BN256 bulk only.
    if swizzle:
        assert mode in ("pipe", "wholeloop") and BLOCK_M == 256 and BLOCK_N == 256, \
            "swizzle requires mode=pipe/wholeloop, BM256/BN256"
    if interleave:
        assert mode == "pipe" and BLOCK_M == 256 and BLOCK_N == 256, \
            "interleave requires mode=pipe, BM256/BN256 (B_COMB, no A_NARROW)"
    if preshuffle_b:
        assert mode == "pipe" and BLOCK_N == 256, \
            "preshuffle_b requires mode=pipe, BN256 (B_COMB); host must run preshuffle_b_mxfp4"
    if b_vgpr:
        assert mode == "pipe" and BLOCK_N == 256 and not interleave and not preshuffle_b, \
            "b_vgpr requires mode=pipe, BN256, not interleave/preshuffle_b; host runs preshuffle_b_vgpr"
    if asm_mma:
        assert mode == "pipe" and BLOCK_N == 256 and not interleave, \
            "asm_mma requires mode=pipe, BN256 (packed scale), not interleave"
    if asm_iter:
        assert mode == "pipe" and BLOCK_M == 256 and BLOCK_N == 256 and not interleave \
            and not asm_mma, \
            "asm_iter (monolithic AGPR cluster) requires pipe, BM256/BN256 (b_vgpr OK for overlap)"
    # 2x4 wave topology: 2 M-waves x 4 N-waves. wave_m = wave_id//4, wave_n = wave_id%4.
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
    LDS_ROW_STRIDE = BPR
    a_lds_size = LDS_BLOCK_M * LDS_ROW_STRIDE  # packed bytes per region buffer
    b_lds_size = LDS_BLOCK_N * LDS_ROW_STRIDE
    K2 = K // 2  # packed-fp4 row stride in bytes

    _WL = (mode == "wholeloop")  # 8-wave whole-loop bare-asm
    _sc_anns = {
        "A_lds_cur_0": fx.Array[fx.Float8E4M3FN, a_lds_size, 16],
        "A_lds_cur_1": fx.Array[fx.Float8E4M3FN, a_lds_size, 16],
        "A_lds_next_0": fx.Array[fx.Float8E4M3FN, a_lds_size, 16],
        "A_lds_next_1": fx.Array[fx.Float8E4M3FN, a_lds_size, 16],
        "B_lds_cur_0": fx.Array[fx.Float8E4M3FN, b_lds_size, 16],
        "B_lds_cur_1": fx.Array[fx.Float8E4M3FN, b_lds_size, 16],
        "B_lds_next_0": fx.Array[fx.Float8E4M3FN, b_lds_size, 16],
        "B_lds_next_1": fx.Array[fx.Float8E4M3FN, b_lds_size, 16],
    }
    # stage3 uses the existing 2-deep cur/next buffers (read-upfront frees the slot for
    # the k+2 g2s; no 3rd buffer -> fits BK256 LDS budget).
    if _WL:
        # scales-in-LDS: 2 ping-pong buffers, each = n_waves(8) x 3 groups (A-r0,A-r1,
        # B-comb) x N_SUB subs x 64 lanes (1 dword). 8-wave n_waves=8 (else wave OOB).
        _SCBUF_8 = 8 * 3 * (BLOCK_K // 128) * 64
        _sc_anns["SC_lds0"] = fx.Array[fx.Int32, _SCBUF_8, 16]
        _sc_anns["SC_lds1"] = fx.Array[fx.Int32, _SCBUF_8, 16]
    SharedStorageFp4Pipe = fx.struct(type("SharedStorageFp4Pipe", (), {"__annotations__": _sc_anns}))

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
        if const_expr(stage3):
            # 2-deep rings indexed by k%2; read-upfront frees slot k%2 for the k+2 g2s.
            s3_ar0 = [a_cur0, a_next0]; s3_ar1 = [a_cur1, a_next1]
            s3_br0 = [b_cur0, b_next0]; s3_br1 = [b_cur1, b_next1]

        lane_id = fx.thread_idx.x % 64
        wave_id = fx.thread_idx.x // 64
        wave_m = wave_id // 4
        wave_n = wave_id % 4
        block_m, block_n = grouped_xcd_pid(fx.block_idx.x, c_m, c_n, BLOCK_M, BLOCK_N,
                                           group_m=group_m, num_xcds=num_xcds, group_n=group_n)

        A0_off = block_m * BLOCK_M * K2
        A1_off = (block_m * BLOCK_M + LDS_BLOCK_M) * K2
        if const_expr(preshuffle_b):
            _BSLAB = 8 * 1024 * N_LDS_STEPS_B  # contiguous (region,k) slab bytes in B_pre
            B_KSTEP = _BSLAB
            B0_off = (block_n * 2 + 0) * KI * _BSLAB
            B1_off = (block_n * 2 + 1) * KI * _BSLAB
        else:
            B_KSTEP = KSTEP
            B0_off = block_n * BLOCK_N * K2
            B1_off = (block_n * BLOCK_N + LDS_BLOCK_N) * K2

        gA = make_fp8_buffer_tensor(A, F8_IR_t)
        gB = make_fp8_buffer_tensor(B_T, F8_IR_t)
        a_div = fx.logical_divide(gA, fx.make_layout(1, 1))
        b_div = fx.logical_divide(gB, fx.make_layout(1, 1))

        mfma = MfmaScaleFp4(N_TILES_A, N_TILES_B, packed=packed_scale)
        _cs = mfma.call_subs_asm if asm_mma else mfma.call_subs  # asm-cluster MFMA (opaque, manual-vmcnt-able) vs intrinsic
        gl_off_a = fp4_g2s_offsets(lane_id, wave_id, K, N_LDS_STEPS_A, BPR, swizzle=swizzle)
        gl_off_b = (fp4_g2s_contiguous_offsets(lane_id, wave_id, N_LDS_STEPS_B) if preshuffle_b
                    else fp4_g2s_offsets(lane_id, wave_id, K, N_LDS_STEPS_B, BPR, swizzle=swizzle))
        gl_off_b_full = fp4_g2s_offsets(lane_id, wave_id, K, N_LDS_STEPS_B_FULL, BPR)
        a_g2s = G2SLoader(a_div, gl_off_a, N_LDS_STEPS_A, F8_IR_t, wave_id)
        b_g2s = G2SLoader(b_div, gl_off_b, N_LDS_STEPS_B, F8_IR_t, wave_id)
        # BN128: combined-128 B G2S (one 128-row G2S fills adjacent b0+b1 halves).
        b_g2s_full = G2SLoader(b_div, gl_off_b_full, N_LDS_STEPS_B_FULL, F8_IR_t, wave_id)
        # B6 BM192/BM128 (A_NARROW): clamp-wave narrow-A G2S. LDS_BLOCK_M < 128 step ->
        # N_LDS_STEPS_A==0 (regular a_g2s loads nothing). Each LDS_BLOCK_M-row A region
        # filled by one combined G2S step where the wave index is clamped to
        # [0, NW_A_ACTIVE-1]; waves >= NW_A_ACTIVE redundantly reload the last 16-row
        # region (same gmem src + LDS dst -> idempotent, in-bounds, det-safe).
        if const_expr(A_NARROW):
            _wid_a = fx.Int32(wave_id)
            eff_wave_a = (_wid_a < fx.Int32(NW_A_ACTIVE)).select(_wid_a, fx.Int32(max(NW_A_ACTIVE - 1, 0)))
            gl_off_a_narrow = fp4_g2s_offsets(lane_id, eff_wave_a, K, 1, BPR)
            a_g2s_narrow = G2SLoader(a_div, gl_off_a_narrow, 1, F8_IR_t, eff_wave_a)
        _asmf = asm_mma or asm_iter or _WL  # asm MFMA needs i32x4 (cbsz:4), not i32x8
        a_s2r = S2RLoaderFp4(wave_m, N_TILES_A, N_SUB, BPR, LDS_ROW_STRIDE, pad=not _asmf, swizzle=swizzle)
        b_s2r = S2RLoaderFp4(wave_n, N_TILES_B, N_SUB, BPR, LDS_ROW_STRIDE, pad=not _asmf, swizzle=swizzle)
        # B->VGPR-direct: skip B's LDS round-trip (no buffer_load_lds, no b ds_read);
        # bvg.load(region, k) buffer_loads each lane's MFMA b-operands straight from the
        # host-preshuffled B (preshuffle_b_vgpr). b_g2s no-op'd; b_read picks the source.
        if const_expr(b_vgpr):
            bvg = BVgprLoader(B_T, block_n, KI, N_TILES_B, N_SUB, c_n * K2, pad=not (asm_mma or asm_iter))
            _bnoop = lambda *a, **k: None
            b_g2s.load = _bnoop
            b_g2s_full.load = _bnoop

        def b_read(region, kk, lds_buf):
            return bvg.load(region, kk) if const_expr(b_vgpr) else b_s2r.load(lds_buf)
        # A-scale num_records must cover the rows the kernel addresses =
        # ceil(c_m/BLOCK_M)*BLOCK_M (the padded extent). ScaleS2R floors
        # dim//group_span, so pass a group_span-ceil'd dim. For aligned M
        # (16*SA_TILES | c_m, true for all BM256 production shapes) this == c_m
        # (byte-identical); only BM192 (16*3=48 ∤ 4096) pads, covering the edge
        # M-tile's scale group (else valid rows 4080-4095 clamp to scale 0 -> SNR 24).
        _sa_q = 16 * SA_TILES
        _sa_qd = ((c_m + _sa_q - 1) // _sa_q) * _sa_q
        if const_expr(packed_scale):
            # PACKED scale: n_tiles E8M0 in ONE i32 (byte t = tile t), opsel per-XDL
            # (4x less scale VMEM vs broadcast). combine_a (python ternary, compile-time)
            # = both M-regions' scale in one coalesced dwordx2 (2 loads/iter not 3).
            sa_s2r = ScaleS2RPackedA2(A_scale, _sa_qd, K, SA_TILES) if (combine_a and not _WL) \
                else ScaleS2RPacked(A_scale, _sa_qd, K, SA_TILES)
            sb_s2r = ScaleBCombPacked(B_scale, c_n, K)
        else:
            sa_s2r = ScaleS2R(A_scale, _sa_qd, K, SA_TILES)
            sb_s2r = ScaleBComb(B_scale, c_n, K) if B_COMB else ScaleBRegion(B_scale, c_n, K, N_TILES_B)
        store_c = StoreCPlain(C, c_m, c_n, mfma.idx, N_TILES_A, N_TILES_B)

        wave_m_offset = wave_m * (N_TILES_A * 16)
        wave_n_offset = wave_n * (N_TILES_B * 16)
        sa_base0 = fx.Int32(block_m * BLOCK_M + wave_m_offset)
        sa_base1 = sa_base0 + fx.Int32(LDS_BLOCK_M)
        sb_base0 = fx.Int32(block_n * BLOCK_N + wave_n_offset)

        if const_expr(_WL):
            # 8-WAVE WHOLE-LOOP bare-asm: entire K-loop = ONE inline-asm hw-loop. occ=2
            # (2 waves/SIMD inherent to the 8-wave WG) hides ds_read latency -> no ring,
            # read-all-upfront + 2-buf ping-pong (a_cur*=buf0, a_next*=buf1). REAL scale
            # via scales-in-LDS (ds_read_b32). 4 quadrants (a0/a1 x b0/b1), 32 VGPR accs.
            K128 = const_expr(K // 128)
            SC_buf = [lds.SC_lds0, lds.SC_lds1]
            _SCW = const_expr(3 * N_SUB * 64)
            rsrc_a = buffer_ops.create_buffer_resource(A, max_size=False, num_records_bytes=c_m * K2)
            rsrc_b = buffer_ops.create_buffer_resource(B_T, max_size=False, num_records_bytes=c_n * K2)
            a_r0 = [a_cur0, a_next0]; a_r1 = [a_cur1, a_next1]
            b_h0 = [b_cur0, b_next0]; b_h1 = [b_cur1, b_next1]
            # prologue: operand buf0 (k=0) + buf1 (k=1) for all 4 regions
            a_g2s.load(a_r0[0], A0_off + 0 * KSTEP)
            a_g2s.load(a_r1[0], A1_off + 0 * KSTEP)
            b_g2s.load(b_h0[0], B0_off + 0 * B_KSTEP)
            b_g2s.load(b_h1[0], B1_off + 0 * B_KSTEP)
            if const_expr(KI > 1):
                a_g2s.load(a_r0[1], A0_off + 1 * KSTEP)
                a_g2s.load(a_r1[1], A1_off + 1 * KSTEP)
                b_g2s.load(b_h0[1], B0_off + 1 * B_KSTEP)
                b_g2s.load(b_h1[1], B1_off + 1 * B_KSTEP)

            def _sc_store(buf, slot, val):
                idx = fx.Int32(wave_id) * fx.Int32(_SCW) + fx.Int32(slot * 64) + lane_id
                pp = fx.add_offset(SC_buf[buf].ptr, fx.make_int_tuple(idx))
                primitive.ptr_store(val, pp)
            for _bk in range_constexpr(2):
                for s in range_constexpr(N_SUB):
                    _k = N_SUB * _bk + s
                    _sc_store(_bk, 0 * N_SUB + s, sa_s2r.load(sa_base0, _k))
                    _sc_store(_bk, 1 * N_SUB + s, sa_s2r.load(sa_base1, _k))
                    _sc_store(_bk, 2 * N_SUB + s, sb_s2r.load(sb_base0, _k))
            # scale stores are ds_write (lgkmcnt); wait_barrier(0) only drains vmcnt.
            _llvm.inline_asm(res=None, operands_=[], asm_string="s_waitcnt lgkmcnt(0)",
                             constraints="", has_side_effects=True)
            wait_barrier(0)

            a0_b = [[a_s2r.base_addr(a_r0[b], s) for s in range_constexpr(N_SUB)] for b in range_constexpr(2)]
            a1_b = [[a_s2r.base_addr(a_r1[b], s) for s in range_constexpr(N_SUB)] for b in range_constexpr(2)]
            b0_b = [[b_s2r.base_addr(b_h0[b], s) for s in range_constexpr(N_SUB)] for b in range_constexpr(2)]
            b1_b = [[b_s2r.base_addr(b_h1[b], s) for s in range_constexpr(N_SUB)] for b in range_constexpr(2)]

            def _gbase(buf):
                v = fx.Int32(fx.ptrtoint(buf.ptr)) + fx.Int32(wave_id) * fx.Int32(1024)
                return rocdl.readfirstlane(T.i32, v)
            ag0 = [_gbase(a_r0[b]) for b in range_constexpr(2)]
            ag1 = [_gbase(a_r1[b]) for b in range_constexpr(2)]
            bg0 = [_gbase(b_h0[b]) for b in range_constexpr(2)]
            bg1 = [_gbase(b_h1[b]) for b in range_constexpr(2)]
            gl_a6 = [fx.Int32(gl_off_a[st]) for st in range_constexpr(N_LDS_STEPS_A)]
            gl_b6 = [fx.Int32(gl_off_b[st]) for st in range_constexpr(N_LDS_STEPS_B)]
            # refill-OTHER (default) is 1-ahead: first refilled k-iter = 1 (not 2). soffset
            # init shifts +AH*KSTEP / scale +AH*N_SUB. refill-same (FP4_WLOTHER=0) = 2-ahead.
            _WLO = int(__import__("os").environ.get("FP4_WLOTHER", "0"))
            AH = const_expr(1 if _WLO else 2)
            soff_a0 = rocdl.readfirstlane(T.i32, A0_off + fx.Int32(AH * KSTEP))
            soff_a1 = rocdl.readfirstlane(T.i32, A1_off + fx.Int32(AH * KSTEP))
            soff_b0 = rocdl.readfirstlane(T.i32, B0_off + fx.Int32(AH * B_KSTEP))
            soff_b1 = rocdl.readfirstlane(T.i32, B1_off + fx.Int32(AH * B_KSTEP))
            sc_rb6 = [fx.ptrtoint(fx.add_offset(SC_buf[b].ptr,
                      fx.make_int_tuple(fx.Int32(wave_id) * fx.Int32(_SCW) + lane_id)))
                      for b in range_constexpr(2)]
            sc_gb6 = [rocdl.readfirstlane(T.i32, fx.Int32(fx.ptrtoint(fx.add_offset(SC_buf[b].ptr,
                      fx.make_int_tuple(fx.Int32(wave_id) * fx.Int32(_SCW))))))
                      for b in range_constexpr(2)]
            sc_voff6 = lane_id * fx.Int32(4)
            def _scsoff_a(base):
                grp = base // fx.Int32(64)
                return rocdl.readfirstlane(T.i32, (grp * fx.Int32(K128) + fx.Int32(AH * N_SUB)) * fx.Int32(256))
            def _scsoff_b(base):
                grp = (base // fx.Int32(256)) * fx.Int32(4) + (base % fx.Int32(256)) // fx.Int32(32)
                return rocdl.readfirstlane(T.i32, (grp * fx.Int32(K128) + fx.Int32(AH * N_SUB)) * fx.Int32(256))
            sc_soff06 = [_scsoff_a(sa_base0), _scsoff_a(sa_base1), _scsoff_b(sb_base0)]

            accs_wl = [mfma.zero_value] * (4 * N_ACCUMS)
            accs_wl = mfma.call_mxfp4_wholeloop_8w(
                a0_b, a1_b, b0_b, b1_b, a_s2r.tile_stride, b_s2r.tile_stride,
                ag0, ag1, bg0, bg1, gl_a6, gl_b6, rsrc_a, rsrc_b,
                fx.Int32(KSTEP), accs_wl, N_SUB, N_LDS_STEPS_A, N_LDS_STEPS_B, fx.Int32(KI),
                soff_a0, soff_a1, soff_b0, soff_b1,
                sc_rb6, sc_gb6, sa_s2r.rsrc, sb_s2r.rsrc, sc_voff6, sc_soff06)
            base_row = block_m * BLOCK_M + wave_m_offset
            base_col = block_n * BLOCK_N + wave_n_offset
            store_c.store(accs_wl[0:N_ACCUMS], base_row + 0, base_col + 0)
            store_c.store(accs_wl[N_ACCUMS:2*N_ACCUMS], base_row + 0, base_col + LDS_BLOCK_N)
            store_c.store(accs_wl[2*N_ACCUMS:3*N_ACCUMS], base_row + LDS_BLOCK_M, base_col + 0)
            store_c.store(accs_wl[3*N_ACCUMS:4*N_ACCUMS], base_row + LDS_BLOCK_M, base_col + LDS_BLOCK_N)
            return

        # A G2S dispatch: BM192/128 (A_NARROW) -> clamp-wave narrow loader; BM256 ->
        # regular a_g2s (byte-identical). const_expr-gated so only one path is traced.
        def a_load(dst, off):
            if const_expr(A_NARROW):
                a_g2s_narrow.load(dst, off)
            else:
                a_g2s.load(dst, off)

        # Per-sub-block scale loaders (K128 index = N_SUB*kiter + s). _sb uses the
        # branch-free `load_halves` so BLOCK_N=128 (per-region) and BLOCK_N=256
        # (combined dwordx4) share one code path; BN256 packing is unchanged.
        # combine-A cache: kiter -> [region0_dword, region1_dword] from the one
        # coalesced dwordx2 load (reg0 issues+caches, reg1 reads).
        _sa_a2_cache = {}

        def _sa(base, kiter, reg=0):
            if const_expr(combine_a):
                # Per K-128 sub-block (N_SUB of them per K-iter): reg0 issues ONE
                # coalesced dwordx2 (both M-regions) + caches by k128 index; reg1 reads
                # the cache (reg0 always precedes reg1 for a given k128).
                out = []
                for s in range_constexpr(N_SUB):
                    k128 = N_SUB * kiter + s
                    if const_expr(reg == 0):
                        if const_expr(k128 not in _sa_a2_cache):
                            _sa_a2_cache[k128] = sa_s2r.load(base, k128)
                        out.append(_sa_a2_cache[k128][0])
                    else:
                        out.append(_sa_a2_cache[k128][1])
                return out
            # packed: each sub-block returns ONE packed i32 (n_tiles E8M0 in 4 bytes).
            return [sa_s2r.load(base, N_SUB * kiter + s) for s in range_constexpr(N_SUB)]

        def _sb(base, kiter):
            if const_expr(packed_scale):
                # ONE packed i32/(k) holds b0 (bytes 0,1) + b1 (bytes 2,3); both
                # N-halves share the dword, carrying the opsel base (0 vs 2) so the
                # call sites stay unchanged (mfma.call_subs reads it when packed).
                sp = [sb_s2r.load(base, N_SUB * kiter + s) for s in range_constexpr(N_SUB)]
                return [(sp[s], 0) for s in range_constexpr(N_SUB)], [(sp[s], 2) for s in range_constexpr(N_SUB)]
            pairs = [sb_s2r.load_halves(base, LDS_BLOCK_N, N_SUB * kiter + s) for s in range_constexpr(N_SUB)]
            return [pairs[s][0] for s in range_constexpr(N_SUB)], [pairs[s][1] for s in range_constexpr(N_SUB)]

        def _quad_il(a_frag, b_frag, c_frag, sa, sb, sides):
            """4wave-style interleave: run one quadrant's N_SUB*N_TILES_A*N_TILES_B
            call_one MFMAs and spread the `sides` closures (G2S load_one steps for the
            k+2 prefetch) evenly into the MFMA stream so the buffer_loads co-issue with
            the matrix unit (hides the G2S-setup MFMA bubble). No intra-quadrant barrier."""
            nmf = N_SUB * N_TILES_A * N_TILES_B
            nsd = len(sides)
            inj = {((t * nmf) // nsd): sides[t] for t in range(nsd)} if nsd else {}
            m = 0
            for s in range_constexpr(N_SUB):
                for i in range_constexpr(N_TILES_A):
                    for j in range_constexpr(N_TILES_B):
                        c_frag[mfma.idx(i, j)] = mfma.call_one(a_frag, b_frag, c_frag, sa, sb, i, j, s)
                        if const_expr(m in inj):
                            inj[m]()
                        m += 1
            return c_frag

        c00_frag = [mfma.zero_value] * N_ACCUMS
        c01_frag = [mfma.zero_value] * N_ACCUMS
        c10_frag = [mfma.zero_value] * N_ACCUMS
        c11_frag = [mfma.zero_value] * N_ACCUMS
        accs = [mfma.zero_value] * (4 * N_ACCUMS)  # asm_iter: flat 32 AGPR accs (c00|c01|c10|c11)
        # 3-stage A LDS ring (reuse the idle B LDS buffers as A's 3rd stage; B is in VGPR
        # via bvg). aring[k%3] always holds A[k]. The A G2S(k+2) writes aring[(k+2)%3] -- a
        # buffer last read at iter k-1 (long done) -> NO WAR with this iter's read -> issue
        # it BEFORE the 64-MFMA cluster so the buffer_load_lds latency overlaps the cluster
        # (the cur/next 24.5% version exposed A G2S after it). Defined unconditionally
        # (plain Python list of already-allocated handles; const_expr-if scopes vars out).
        _ASM_RING = asm_iter and b_vgpr
        aring0 = [a_cur0, a_next0, b_cur0]
        aring1 = [a_cur1, a_next1, b_cur1]
        if const_expr(_ASM_RING):
            a_load(aring0[0], A0_off + 0 * KSTEP)
            a_load(aring1[0], A1_off + 0 * KSTEP)
            a_load(aring0[1], A0_off + 1 * KSTEP)
            a_load(aring1[1], A1_off + 1 * KSTEP)
            sa0 = _sa(sa_base0, 0)
            sa1 = _sa(sa_base1, 0, 1)
            sb0, sb1 = _sb(sb_base0, 0)
            wait_barrier(0)  # A0,A1 landed + cross-wave visible before loop reads

        if const_expr(stage3):
            # 3-stage prologue: G2S k=0 -> ring slot0, k=1 -> ring slot1 (2-iter-ahead).
            a_load(s3_ar0[0], A0_off + 0 * KSTEP); a_load(s3_ar1[0], A1_off + 0 * KSTEP)
            b_g2s.load(s3_br0[0], B0_off + 0 * B_KSTEP); b_g2s.load(s3_br1[0], B1_off + 0 * B_KSTEP)
            a_load(s3_ar0[1], A0_off + 1 * KSTEP); a_load(s3_ar1[1], A1_off + 1 * KSTEP)
            b_g2s.load(s3_br0[1], B0_off + 1 * B_KSTEP); b_g2s.load(s3_br1[1], B1_off + 1 * B_KSTEP)
            wait_barrier(0)  # drain both iters' G2S + cross-wave s_barrier (slots 0,1 visible)
            sa0 = _sa(sa_base0, 0); sa1 = _sa(sa_base1, 0, 1)
            sb0, sb1 = _sb(sb_base0, 0)

        if const_expr(not _ASM_RING and not stage3):
            if const_expr(B_COMB):
                b_g2s.load(b_cur0, B0_off + 0 * B_KSTEP)
            else:
                b_g2s_full.load(b_cur0, B0_off + 0 * B_KSTEP)  # 128-row combined -> b_cur0+b_cur1
            a_load(a_cur0, A0_off + 0 * KSTEP)
            if const_expr(B_COMB):
                b_g2s.load(b_cur1, B1_off + 0 * B_KSTEP)
            a_load(a_cur1, A1_off + 0 * KSTEP)

            if wave_m == 1:
                rocdl.s_barrier()

            if const_expr(B_COMB):
                wait_barrier(N_LDS_STEPS_A + N_LDS_STEPS_B)
            else:
                wait_barrier(0)  # BN128: conservative prologue drain (one-time, correctness-first)

            if const_expr(B_COMB):
                b_g2s.load(b_next0, B0_off + 1 * B_KSTEP)
            else:
                b_g2s_full.load(b_next0, B0_off + 1 * B_KSTEP)  # 128-row combined -> b_next0+b_next1
            a_load(a_next0, A0_off + 1 * KSTEP)
            if const_expr(B_COMB):
                b_g2s.load(b_next1, B1_off + 1 * B_KSTEP)

            if const_expr(B_COMB):
                wait_barrier(N_LDS_STEPS_A + 2 * N_LDS_STEPS_B)
            else:
                wait_barrier(0)  # BN128: conservative prologue drain

            sa0 = _sa(sa_base0, 0)
            sa1 = _sa(sa_base1, 0, 1)
            sb0, sb1 = _sb(sb_base0, 0)

        for k in range_constexpr(KI - 2):
            if const_expr(stage3):
                # ===== gluon-style 3-stage / 1-barrier body (BK128, 3-deep ring) =====
                # entry invariant: ring slots k%3,(k+1)%3 filled+visible; sa*/sb*=scales[k].
                # 2-buffer read-upfront: slot rk=k%2 holds tile k. Read it to registers,
                # MFMA (consumes -> slot rk dead), ONE barrier (cross-wave: all done), then
                # g2s k+2 reuses slot rk. tile k+1 lives in slot (k+1)%2 (untouched this iter).
                rk = k % 2
                ar0, ar1 = s3_ar0[rk], s3_ar1[rk]
                br0, br1 = s3_br0[rk], s3_br1[rk]
                sa0n = _sa(sa_base0, k + 1); sa1n = _sa(sa_base1, k + 1, 1)
                sb0n, sb1n = _sb(sb_base0, k + 1)
                a0_frag = a_s2r.load(ar0); a1_frag = a_s2r.load(ar1)
                b0_frag = b_read(0, k, br0); b1_frag = b_read(1, k, br1)
                rocdl.s_setprio(1)
                c00_frag = _cs(a0_frag, b0_frag, c00_frag, sa0, sb0, N_SUB)
                c01_frag = _cs(a0_frag, b1_frag, c01_frag, sa0, sb1, N_SUB)
                c10_frag = _cs(a1_frag, b0_frag, c10_frag, sa1, sb0, N_SUB)
                c11_frag = _cs(a1_frag, b1_frag, c11_frag, sa1, sb1, N_SUB)
                rocdl.s_setprio(0)
                if const_expr(int(__import__("os").environ.get("FP4_SCHED", "0"))):
                    _nm = 4 * N_ACCUMS * N_SUB; _nd = 2 * (N_TILES_A + N_TILES_B) * N_SUB
                    _b = _nm // _nd; _ex = _nm % _nd; _mi = 0
                    for _j in range_constexpr(_nd):
                        _g = _b + (1 if _j < _ex else 0)
                        if const_expr(_g > 0):
                            rocdl.sched_mfma(_g); _mi += _g
                        rocdl.sched_dsrd(1)
                    if const_expr(_mi < _nm):
                        rocdl.sched_mfma(_nm - _mi)
                rocdl.s_barrier()  # cross-wave: all waves done reading slot rk before g2s
                a_load(ar0, A0_off + (k + 2) * KSTEP); a_load(ar1, A1_off + (k + 2) * KSTEP)
                b_g2s.load(br0, B0_off + (k + 2) * B_KSTEP); b_g2s.load(br1, B1_off + (k + 2) * B_KSTEP)
                wait_barrier(0)  # drain g2s vmcnt + cross-wave visibility for iter k+2 read
                sa0, sa1 = sa0n, sa1n
                sb0, sb1 = sb0n, sb1n
                continue
            if const_expr(_ASM_RING):
                # 3-stage-ring overlap: A[k+2] G2S + B[k+1] VGPR-load issued BEFORE the
                # 64-MFMA cluster so both buffer_load latencies hide under the MFMA. The
                # single wait_barrier(0) after the cluster is then cheap (loads already
                # in flight during the 64 MFMAs). aring[(k+2)%3] is WAR-free (last read
                # at iter k-1, globally done via the prev iter's wait_barrier s_barrier).
                sa0n = _sa(sa_base0, k + 1)
                sb0n, sb1n = _sb(sb_base0, k + 1)
                sa1n = _sa(sa_base1, k + 1, 1)
                rd0, rd1 = aring0[k % 3], aring1[k % 3]
                wr0, wr1 = aring0[(k + 2) % 3], aring1[(k + 2) % 3]
                a0_frag = a_s2r.load(rd0)                         # ds_read A[k] (lgkmcnt; dep -> auto-wait at cluster)
                a1_frag = a_s2r.load(rd1)
                b0_frag = bvg.load(0, k)                          # B[k] -> VGPR single-buffer (JIT)
                b1_frag = bvg.load(1, k)
                a_load(wr0, A0_off + (k + 2) * KSTEP)             # A[k+2] -> fresh LDS (overlaps cluster)
                a_load(wr1, A1_off + (k + 2) * KSTEP)
                rocdl.s_setprio(1)
                accs = mfma.call_iter_asm(a0_frag, a1_frag, b0_frag, b1_frag,
                                          sa0, sa1, sb0, sb1, accs, N_SUB)
                rocdl.s_setprio(0)
                wait_barrier(0)  # drain A prefetch (overlapped cluster -> cheap) + s_barrier (vis/WAR)
                sa0, sa1 = sa0n, sa1n
                sb0, sb1 = sb0n, sb1n
                continue
            if const_expr(asm_iter):
                # MONOLITHIC B-LDS path (b_vgpr off): ONE 64-MFMA asm cluster, 32 accs AGPR
                # tied across iters, k+2 G2S prefetch (B stays LDS for reuse).
                sa0n = _sa(sa_base0, k + 1)
                sb0n, sb1n = _sb(sb_base0, k + 1)
                sa1n = _sa(sa_base1, k + 1, 1)
                wait_barrier(0)  # drain prev-iter prefetch BEFORE reads; +s_barrier sync
                a0_frag = a_s2r.load(a_cur0)
                a1_frag = a_s2r.load(a_cur1)
                b0_frag = b_read(0, k, b_cur0)
                b1_frag = b_read(1, k, b_cur1)
                rocdl.s_barrier()  # all waves read cur into frags before prefetch overwrites cur (WAR)
                rocdl.s_setprio(1)
                accs = mfma.call_iter_asm(a0_frag, a1_frag, b0_frag, b1_frag,
                                          sa0, sa1, sb0, sb1, accs, N_SUB)
                rocdl.s_setprio(0)
                rocdl.s_barrier()  # all waves done ds-reading cur (cluster lgkmcnt-consumed) before overwrite
                b_g2s.load(b_cur0, B0_off + (k + 2) * B_KSTEP)
                a_load(a_cur0, A0_off + (k + 2) * KSTEP)
                b_g2s.load(b_cur1, B1_off + (k + 2) * B_KSTEP)
                a_load(a_next1, A1_off + (k + 1) * KSTEP)  # asymmetric (required; symmetric worse)
                wait_barrier(0)  # drain A prefetch
                a_cur0, a_next0 = a_next0, a_cur0
                a_cur1, a_next1 = a_next1, a_cur1
                b_cur0, b_next0 = b_next0, b_cur0
                b_cur1, b_next1 = b_next1, b_cur1
                sa0, sa1 = sa0n, sa1n
                sb0, sb1 = sb0n, sb1n
                continue
            if const_expr(interleave):
                # 4wave-style interleave: per-quadrant wait_barrier + k+2 G2S buffer_loads
                # spread into the call_one MFMA stream (co-issue -> hides G2S-setup bubble).
                # FP4_IL_VMC env tunes the per-quadrant vmcnt drain (0 = safest full drain;
                # raise to leave G2S in flight once det-clean). BN256 bulk only.
                _ilvmc = const_expr(int(__import__("os").environ.get("FP4_IL_VMC", "0")))
                sa0n = _sa(sa_base0, k + 1)
                sb0n, sb1n = _sb(sb_base0, k + 1)
                sa1n = _sa(sa_base1, k + 1, 1)
                # c00 (a0xb0); spread a_next1 G2S
                b0_frag = b_s2r.load(b_cur0)
                a0_frag = a_s2r.load(a_cur0)
                wait_barrier(_ilvmc)
                b1_frag = b_s2r.load(b_cur1)
                rocdl.s_setprio(1)
                c00_frag = _quad_il(a0_frag, b0_frag, c00_frag, sa0, sb0,
                                    [(lambda st=st: a_g2s.load_one(a_next1, A1_off + (k + 1) * KSTEP, st))
                                     for st in range(N_LDS_STEPS_A)])
                rocdl.s_setprio(0)
                # c01 (a0xb1); spread b_cur0(k+2) G2S
                wait_barrier(_ilvmc)
                rocdl.s_setprio(1)
                c01_frag = _quad_il(a0_frag, b1_frag, c01_frag, sa0, sb1,
                                    [(lambda st=st: b_g2s.load_one(b_cur0, B0_off + (k + 2) * B_KSTEP, st))
                                     for st in range(N_LDS_STEPS_B)])
                rocdl.s_setprio(0)
                # c10 (a1xb0); spread a_cur0(k+2) G2S
                a1_frag = a_s2r.load(a_cur1)
                wait_barrier(_ilvmc)
                rocdl.s_setprio(1)
                c10_frag = _quad_il(a1_frag, b0_frag, c10_frag, sa1, sb0,
                                    [(lambda st=st: a_g2s.load_one(a_cur0, A0_off + (k + 2) * KSTEP, st))
                                     for st in range(N_LDS_STEPS_A)])
                rocdl.s_setprio(0)
                # c11 (a1xb1); spread b_cur1(k+2) G2S
                wait_barrier(_ilvmc)
                rocdl.s_setprio(1)
                c11_frag = _quad_il(a1_frag, b1_frag, c11_frag, sa1, sb1,
                                    [(lambda st=st: b_g2s.load_one(b_cur1, B1_off + (k + 2) * B_KSTEP, st))
                                     for st in range(N_LDS_STEPS_B)])
                rocdl.s_setprio(0)
                a_cur0, a_next0 = a_next0, a_cur0
                a_cur1, a_next1 = a_next1, a_cur1
                b_cur0, b_next0 = b_next0, b_cur0
                b_cur1, b_next1 = b_next1, b_cur1
                sa0, sa1 = sa0n, sa1n
                sb0, sb1 = sb0n, sb1n
                continue
            sa0n = _sa(sa_base0, k + 1)
            # FP4_LEANBAR: speed-ceiling probe (aiter has ~8 barriers TOTAL vs our 8/iter).
            # Strip the 7 plain s_barriers, keep only the 1 vmcnt-drain wait_barrier.
            # Correctness IGNORED here (measure perf first; fix races only if speed wins).
            _LEANBAR = const_expr(__import__("os").environ.get("FP4_LEANBAR", "0") == "1")

            b0_frag = b_read(0, k, b_cur0)
            a0_frag = a_s2r.load(a_cur0)
            a_load(a_next1, A1_off + (k + 1) * KSTEP)
            if const_expr(not _LEANBAR):
                rocdl.s_barrier()

            # SW-pipeline: issue b_cur1 ds_read INSIDE the c00 region (after the
            # barrier that guarantees b_cur1's prev-iter write is visible) so its
            # LDS-read latency overlaps the c00 MFMA (attacks the 34% MFMA-dep
            # stall). b_cur1 isn't overwritten until the k+2 g2s (4 barriers later).
            b1_frag = b_read(1, k, b_cur1)
            rocdl.s_setprio(1)
            c00_frag = _cs(a0_frag, b0_frag, c00_frag, sa0, sb0, N_SUB)
            rocdl.s_setprio(0)
            if const_expr(not _LEANBAR):
                rocdl.s_barrier()

            if const_expr(B_COMB):
                b_g2s.load(b_cur0, B0_off + (k + 2) * B_KSTEP)
            sb0n, sb1n = _sb(sb_base0, k + 1)
            if const_expr(not _LEANBAR):
                rocdl.s_barrier()

            rocdl.s_setprio(1)
            c01_frag = _cs(a0_frag, b1_frag, c01_frag, sa0, sb1, N_SUB)
            rocdl.s_setprio(0)
            if const_expr(not _LEANBAR):
                rocdl.s_barrier()

            a1_frag = a_s2r.load(a_cur1)
            a_load(a_cur0, A0_off + (k + 2) * KSTEP)
            sa1n = _sa(sa_base1, k + 1, 1)
            if const_expr(not _LEANBAR):
                rocdl.s_barrier()

            rocdl.s_setprio(1)
            c10_frag = _cs(a1_frag, b0_frag, c10_frag, sa1, sb0, N_SUB)
            rocdl.s_setprio(0)
            if const_expr(not _LEANBAR):
                rocdl.s_barrier()

            if const_expr(B_COMB):
                b_g2s.load(b_cur1, B1_off + (k + 2) * B_KSTEP)
                if const_expr(not _LEANBAR):
                    wait_barrier(2 * N_LDS_STEPS_A + N_LDS_STEPS_B)
            else:
                # BN128 combined k+2 (b_cur0+b_cur1). WAR-safe: b0/b1 reads above are
                # consumed by the c00/c01/c10 mmas before this overwrite.
                b_g2s_full.load(b_cur0, B0_off + (k + 2) * B_KSTEP)
                # r_k7: wait(3) raced (r_k6); try the tightest overlap-allowing drain = 1
                # outstanding (combined-spill writes nearly fully drained, but A-refills can
                # overlap). det0-gated: if this races too, the merged-spill needs vmcnt(0)
                # and pipe-BN128 perf is infeasible (conservative ≈ staged = loses).
                wait_barrier(1)

            rocdl.s_setprio(1)
            c11_frag = _cs(a1_frag, b1_frag, c11_frag, sa1, sb1, N_SUB)
            rocdl.s_setprio(0)
            if const_expr(not _LEANBAR):
                rocdl.s_barrier()

            # FP4_SCHED: gluon LLIR-scheduler mechanism. With LEANBAR (barrier-free body =
            # one schedulable region), emit sched_group_barrier cadence to interleave the
            # ds_read (operand LR) EVENLY into the mfma stream -> bury LDS-read latency in
            # the MFMA shadow (the 4274->5253 gluon lever is exactly this scheduler pass).
            # totals: mfma = 4*N_ACCUMS*N_SUB; dsrd = 2*(NTA+NTB)*N_SUB (A0/A1 + B0/B1 reads).
            _SCHED = const_expr(int(__import__("os").environ.get("FP4_SCHED", "0")))
            if const_expr(_SCHED):  # s_barrier is a runtime barrier; LLVM scheduler can still
                                    # reorder ds_read across it (no data dep) per the sched groups
                _nm = 4 * N_ACCUMS * N_SUB
                _nd = 2 * (N_TILES_A + N_TILES_B) * N_SUB
                _base = _nm // _nd; _extra = _nm % _nd; _mi = 0
                for _j in range_constexpr(_nd):
                    _g = _base + (1 if _j < _extra else 0)
                    if const_expr(_g > 0):
                        rocdl.sched_mfma(_g); _mi += _g
                    rocdl.sched_dsrd(1)
                if const_expr(_mi < _nm):
                    rocdl.sched_mfma(_nm - _mi)

            a_cur0, a_next0 = a_next0, a_cur0
            a_cur1, a_next1 = a_next1, a_cur1
            b_cur0, b_next0 = b_next0, b_cur0
            b_cur1, b_next1 = b_next1, b_cur1
            sa0, sa1 = sa0n, sa1n
            sb0, sb1 = sb0n, sb1n

        if const_expr(asm_iter):
            # asm_iter tail: 2 final clusters (KI-2, KI-1), no prefetch; then demux+store.
            for _kt in range_constexpr(2):
                kk = (KI - 2) + _kt
                if const_expr(_ASM_RING):
                    # A[KI-2],A[KI-1] already in the ring (prefetched by main loop's last
                    # 2 iters). No prefetch; just read aring[kk%3].
                    a0_frag = a_s2r.load(aring0[kk % 3])
                    a1_frag = a_s2r.load(aring1[kk % 3])
                    b0_frag = bvg.load(0, kk)
                    b1_frag = bvg.load(1, kk)
                else:
                    wait_barrier(0)  # drain prev prefetch BEFORE reads
                    a0_frag = a_s2r.load(a_cur0)
                    a1_frag = a_s2r.load(a_cur1)
                    b0_frag = b_read(0, kk, b_cur0)
                    b1_frag = b_read(1, kk, b_cur1)
                    if const_expr(_kt == 0):
                        a_load(a_next1, A1_off + (KI - 1) * KSTEP)
                rocdl.s_setprio(1)
                accs = mfma.call_iter_asm(a0_frag, a1_frag, b0_frag, b1_frag,
                                          sa0, sa1, sb0, sb1, accs, N_SUB)
                rocdl.s_setprio(0)
                rocdl.s_barrier()
                if const_expr(_kt == 0):
                    if const_expr(not _ASM_RING):
                        a_cur0, a_next0 = a_next0, a_cur0
                        a_cur1, a_next1 = a_next1, a_cur1
                        b_cur0, b_next0 = b_next0, b_cur0
                        b_cur1, b_next1 = b_next1, b_cur1
                    sa0, sa1 = _sa(sa_base0, KI - 1), _sa(sa_base1, KI - 1, 1)
                    sb0, sb1 = _sb(sb_base0, KI - 1)
            base_row = block_m * BLOCK_M + wave_m_offset
            base_col = block_n * BLOCK_N + wave_n_offset
            store_c.store(accs[0:N_ACCUMS], base_row + 0, base_col + 0)
            store_c.store(accs[N_ACCUMS:2*N_ACCUMS], base_row + 0, base_col + LDS_BLOCK_N)
            store_c.store(accs[2*N_ACCUMS:3*N_ACCUMS], base_row + LDS_BLOCK_M, base_col + 0)
            store_c.store(accs[3*N_ACCUMS:4*N_ACCUMS], base_row + LDS_BLOCK_M, base_col + LDS_BLOCK_N)
        elif const_expr(stage3):
            # 3-stage tail: last 2 iters (KI-2,KI-1) already in ring (loop's last k+2 G2S).
            for _kt in range_constexpr(2):
                kk = (KI - 2) + _kt; rk = kk % 2
                a0_frag = a_s2r.load(s3_ar0[rk]); a1_frag = a_s2r.load(s3_ar1[rk])
                b0_frag = b_read(0, kk, s3_br0[rk]); b1_frag = b_read(1, kk, s3_br1[rk])
                rocdl.s_setprio(1)
                c00_frag = _cs(a0_frag, b0_frag, c00_frag, sa0, sb0, N_SUB)
                c01_frag = _cs(a0_frag, b1_frag, c01_frag, sa0, sb1, N_SUB)
                c10_frag = _cs(a1_frag, b0_frag, c10_frag, sa1, sb0, N_SUB)
                c11_frag = _cs(a1_frag, b1_frag, c11_frag, sa1, sb1, N_SUB)
                rocdl.s_setprio(0)
                rocdl.s_barrier()
                if const_expr(_kt == 0):
                    sa0, sa1 = _sa(sa_base0, KI - 1), _sa(sa_base1, KI - 1, 1)
                    sb0, sb1 = _sb(sb_base0, KI - 1)
            base_row = block_m * BLOCK_M + wave_m_offset
            base_col = block_n * BLOCK_N + wave_n_offset
            store_c.store(c00_frag, base_row + 0, base_col + 0)
            store_c.store(c01_frag, base_row + 0, base_col + LDS_BLOCK_N)
            store_c.store(c10_frag, base_row + LDS_BLOCK_M, base_col + 0)
            store_c.store(c11_frag, base_row + LDS_BLOCK_M, base_col + LDS_BLOCK_N)
        else:
            # Step k = K_ITERS - 2 (prefetch last iter)
            k = KI - 2
            sa0n = _sa(sa_base0, KI - 1)
            sa1n = _sa(sa_base1, KI - 1, 1)
            sb0n, sb1n = _sb(sb_base0, KI - 1)

            b0_frag = b_read(0, KI - 2, b_cur0)
            a0_frag = a_s2r.load(a_cur0)
            rocdl.s_barrier()

            rocdl.s_setprio(1)
            c00_frag = _cs(a0_frag, b0_frag, c00_frag, sa0, sb0, N_SUB)
            rocdl.s_setprio(0)
            rocdl.s_barrier()

            b1_frag = b_read(1, KI - 2, b_cur1)
            rocdl.s_barrier()

            rocdl.s_setprio(1)
            c01_frag = _cs(a0_frag, b1_frag, c01_frag, sa0, sb1, N_SUB)
            rocdl.s_setprio(0)
            rocdl.s_barrier()

            a1_frag = a_s2r.load(a_cur1)
            a_load(a_next1, A1_off + (KI - 1) * KSTEP)
            rocdl.s_barrier()

            rocdl.s_setprio(1)
            c10_frag = _cs(a1_frag, b0_frag, c10_frag, sa1, sb0, N_SUB)
            rocdl.s_setprio(0)
            rocdl.s_barrier()

            b0_frag = b_read(0, KI - 1, b_next0)
            rocdl.s_barrier()

            rocdl.s_setprio(1)
            c11_frag = _cs(a1_frag, b1_frag, c11_frag, sa1, sb1, N_SUB)
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
            c00_frag = _cs(a0_frag, b0_frag, c00_frag, sa0, sb0, N_SUB)
            rocdl.s_setprio(0)
            rocdl.s_barrier()

            b1_frag = b_read(1, KI - 1, b_cur1)
            rocdl.s_barrier()

            rocdl.s_setprio(1)
            c01_frag = _cs(a0_frag, b1_frag, c01_frag, sa0, sb1, N_SUB)
            rocdl.s_setprio(0)
            rocdl.s_barrier()

            a1_frag = a_s2r.load(a_cur1)
            rocdl.s_barrier()

            rocdl.s_setprio(1)
            c10_frag = _cs(a1_frag, b0_frag, c10_frag, sa1, sb0, N_SUB)
            c11_frag = _cs(a1_frag, b1_frag, c11_frag, sa1, sb1, N_SUB)
            rocdl.s_setprio(0)
            rocdl.s_barrier()

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
        kernel_gemm_pipe(
            A, B_T, C, A_scale, B_scale, c_m, c_n,
            value_attrs={"rocdl.waves_per_eu": 2, "rocdl.flat_work_group_size": "512,512"},
        ).launch(grid=(grid_x, 1, 1), block=(512, 1, 1), stream=stream)

    return launch_gemm
