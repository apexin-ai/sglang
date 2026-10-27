# SPDX-License-Identifier: Apache-2.0
# Ported from DeepInfer's sm90_ops/indexer.py (decode kernels only), which adapts the e2m1 decode and the
# paged FP4 index-K page access of this package's fp4_indexer.py and the FP4 -> FP8 re-encoding proposed
# in sgl-project/sglang#39086.
# Triton JIT signatures stay unannotated (Triton reads parameter annotations for specialization).
"""SM90 (Hopper) decode indexer logits: Triton stand-ins for DeepGEMM's SM100-only
``fp8_fp4_paged_mqa_logits`` and ``fp8_fp4_paged_sparse_mqa_logits``.

Inputs are the DeepGEMM FP4 formats: index Q packed e2m1 ``int8 [T, H, 64]`` (low nibble = even element)
with packed UE8M0 ``int32 [T, H]`` scales (byte g scales elements ``[32g, 32g + 32)``), and the paged
index-K cache ``uint8 [pages, page, 1, 68]`` with a page-wide ``[page * 64 data | page * 4 sf]`` split.

Numerics follow the reference indexer of the DeepSeek V4.1 inference code (``Indexer.forward``)::

    s_h = bf16(q_h . k)                          # q, k = FP4 values with power-of-two per-32 scales
    out = bf16(sum_h bf16(relu(s_h) * w_h))      # head sum accumulated in fp32

The 128-wide dot is four exact 32-wide FP8 tensor-core dots summed in fp32 as ``((p0 + p1) + p2) + p3``.
Index Q rows are re-encoded once as E4M3 values ``e2m1 * 2**(E_g - E_max + 6)`` with the power-of-two row
factor ``2**(E_max - 133)`` folded into the head weight; paged K is decoded per block to unscaled E4M3 and
its block scale multiplies the exact 32-wide dot. The re-encoding is exact (bitwise equal to the reference)
when every nonzero Q block of a row has ``E_max - E_g <= 14``; Hadamard-rotated index Q keeps block scales
within a few binades, and a block further below falls into the E4M3 subnormal range (relative error per
element at most ``2**-16`` of the row's largest scale).

Output contracts (as DeepGEMM; columns left unwritten stay uninitialized, so consumers bound reads by the
valid length):

* ``paged_mqa_logits``: ``[B, max_seqlen_k]`` bf16, column ``j`` <-> position ``j``, written for ``j < klen``.
* ``paged_sparse_mqa_logits``: ``[B, M * bs]`` bf16, column ``j`` <-> position ``cand[j // bs] * bs + j % bs``,
  written for the first ``min(M, ceil(klen / bs))`` slots; positions ``>= klen`` or of a negative block
  are ``-inf``.

Rows are 1024-byte aligned (DeepSelect requires it). Launch shapes depend only on tensor shapes, so both
entry points are CUDA-graph safe.
"""

import functools
from typing import NamedTuple

import torch
import triton
import triton.language as tl

INDEX_HEAD_DIM = 128
_ROW_ALIGN_ELEMS = 512  # 1024-byte bf16 rows, as DeepGEMM / deep_select expect


@functools.cache
def _num_sms(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


class PreparedQ(NamedTuple):
    """Index Q re-encoded for the SM90 kernels."""

    q8: torch.Tensor  # [T, H, 128] float8_e4m3fn, per-(token, head) shifted e2m1 values
    wq: torch.Tensor  # [T, H] fp32, head weight * 2**(E_max - 133)


# ----------------------------------------------------------------------------------------------
# Shared device helpers
# ----------------------------------------------------------------------------------------------


@triton.jit
def _e2m1_to_f32(code):
    """code: int32 nibble in [0, 15] (bit 3 sign, bits 2..1 exponent, bit 0 mantissa)."""
    e = (code >> 1) & 3
    m = (code & 1).to(tl.float32)
    mag = tl.where(e == 0, m * 0.5, (1.0 + 0.5 * m) * tl.where(e == 3, 4.0, tl.where(e == 2, 2.0, 1.0)))
    return tl.where((code & 8) != 0, -mag, mag)


@triton.jit
def _pow2_f32(e):
    """2**e for an int32 tensor e <= 127; 0 below the fp32 normal range (never reached by real rows)."""
    bits = (tl.maximum(e, -126) + 127) << 23
    return tl.where(e >= -126, bits.to(tl.float32, bitcast=True), 0.0)


@triton.jit
def _sf_bytes(sf):
    """Split packed UE8M0 int32 scale factors into their four exponent bytes and the row maximum."""
    e0 = sf & 0xFF
    e1 = (sf >> 8) & 0xFF
    e2 = (sf >> 16) & 0xFF
    e3 = (sf >> 24) & 0xFF
    emax = tl.maximum(tl.maximum(e0, e1), tl.maximum(e2, e3))
    return e0, e1, e2, e3, emax


@triton.jit
def _decode_fp4_block(pay, eg, emax):
    """pay: [R, 16] int32 bytes of one 32-element block; eg/emax: [R] exponents -> [R, 32] fp32 shifted values."""
    shift = _pow2_f32(eg - emax + 6)[:, None]
    lo = _e2m1_to_f32(pay & 15) * shift
    hi = _e2m1_to_f32((pay >> 4) & 15) * shift
    return tl.interleave(lo, hi)


@triton.jit
def _e2m1x2_to_e4m3(packed):
    """uint8 bytes holding two e2m1 codes -> (low, high) uint8 E4M3 bit patterns of the *unscaled* values.

    Exact (every e2m1 value is an E4M3 value). Four bytes per 32-bit register: a byte-permute LUT maps the
    3-bit magnitudes {0, .5, 1, 1.5, 2, 3, 4, 6} to {0x00, 0x30, 0x38, 0x3C, 0x40, 0x44, 0x48, 0x4C} and
    the sign bit is or-ed in. The mapping is per byte, so it does not depend on how Triton packs lanes."""
    return tl.inline_asm_elementwise(
        asm="""
        {
        .reg .b32 t, u, sel, lut_lo, lut_hi, sgn;
        mov.b32 lut_lo, 0x3C383000;
        mov.b32 lut_hi, 0x4C484440;
        and.b32 t, $2, 0x07070707;
        shr.b32 u, t, 12;
        or.b32 sel, t, u;
        prmt.b32 t, lut_lo, lut_hi, sel;
        prmt.b32 t, t, t, 0x3120;
        shl.b32 sgn, $2, 4;
        and.b32 sgn, sgn, 0x80808080;
        or.b32 $0, t, sgn;
        shr.b32 t, $2, 4;
        and.b32 t, t, 0x07070707;
        shr.b32 u, t, 12;
        or.b32 sel, t, u;
        prmt.b32 t, lut_lo, lut_hi, sel;
        prmt.b32 t, t, t, 0x3120;
        and.b32 sgn, $2, 0x80808080;
        or.b32 $1, t, sgn;
        }
        """,
        constraints="=r,=r,r",
        args=[packed],
        dtype=(tl.uint8, tl.uint8),
        is_pure=True,
        pack=4,
    )


@triton.jit
def _fp4_block_e4m3(pay, BN: tl.constexpr):
    """[BN, 16] uint8 (32 packed e2m1 codes per row) -> [BN, 32] float8e4nv unscaled values, natural order."""
    lo, hi = _e2m1x2_to_e4m3(pay)
    return tl.join(lo, hi).reshape(BN, 32).to(tl.float8e4nv, bitcast=True)


# ----------------------------------------------------------------------------------------------
# Prepass: FP4 rows -> shifted E4M3 rows + power-of-two row factor
# ----------------------------------------------------------------------------------------------


@triton.jit
def _fp4_rows_to_fp8_kernel(
    x_ptr,  # [R, 64] int8 packed e2m1
    sf_ptr,  # [R] int32 packed ue8m0
    w_ptr,  # [T, H] bf16 or fp32 head weights (row r = t * H + h); unused when not HAS_W
    out_ptr,  # [R, 128] e4m3
    scale_ptr,  # [R] fp32
    R,
    H,
    stride_wt,
    HAS_W: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    rmask = rows < R
    rows64 = rows.to(tl.int64)
    sf = tl.load(sf_ptr + rows64, mask=rmask, other=0)
    e0, e1, e2, e3, emax = _sf_bytes(sf)
    b16 = tl.arange(0, 16)
    b32 = tl.arange(0, 32)
    xrow = x_ptr + rows64[:, None] * 64 + b16[None, :]
    orow = out_ptr + rows64[:, None] * 128 + b32[None, :]
    rm = rmask[:, None]
    odt = out_ptr.dtype.element_ty
    tl.store(orow, _decode_fp4_block(tl.load(xrow, mask=rm, other=0).to(tl.int32) & 0xFF, e0, emax).to(odt), mask=rm)
    tl.store(orow + 32, _decode_fp4_block(tl.load(xrow + 16, mask=rm, other=0).to(tl.int32) & 0xFF, e1, emax).to(odt), mask=rm)
    tl.store(orow + 64, _decode_fp4_block(tl.load(xrow + 32, mask=rm, other=0).to(tl.int32) & 0xFF, e2, emax).to(odt), mask=rm)
    tl.store(orow + 96, _decode_fp4_block(tl.load(xrow + 48, mask=rm, other=0).to(tl.int32) & 0xFF, e3, emax).to(odt), mask=rm)
    scale = _pow2_f32(emax - 133)
    if HAS_W:
        t = (rows // H).to(tl.int64)
        w = tl.load(w_ptr + t * stride_wt + rows % H, mask=rmask, other=0.0).to(tl.float32)
        scale = scale * w
    tl.store(scale_ptr + rows64, scale, mask=rmask)


def _fp4_rows_to_fp8(x: torch.Tensor, sf: torch.Tensor, weights: torch.Tensor | None, num_heads: int) -> tuple[torch.Tensor, torch.Tensor]:
    rows = sf.numel()
    assert x.dtype == torch.int8 and x.is_contiguous() and x.numel() == rows * 64, f"{x.shape=} {x.dtype=}"
    assert sf.dtype == torch.int32 and sf.is_contiguous(), f"{sf.shape=} {sf.dtype=}"
    out = torch.empty((rows, INDEX_HEAD_DIM), dtype=torch.float8_e4m3fn, device=x.device)
    scale = torch.empty((rows,), dtype=torch.float32, device=x.device)
    if rows > 0:
        if weights is not None:
            assert weights.dtype in (torch.bfloat16, torch.float32) and weights.stride(-1) == 1 and weights.shape == (rows // num_heads, num_heads)
        block_r = 32
        _fp4_rows_to_fp8_kernel[(triton.cdiv(rows, block_r),)](
            x,
            sf,
            weights if weights is not None else scale,
            out,
            scale,
            rows,
            num_heads,
            weights.stride(0) if weights is not None else 0,
            HAS_W=weights is not None,
            BLOCK_R=block_r,
            num_warps=4,
        )
    return out, scale


def prepare_q(q_fp4: torch.Tensor, q_sf: torch.Tensor, weights: torch.Tensor) -> PreparedQ:
    """q_fp4 int8 [T, H, 64], q_sf int32 [T, H], weights bf16 [T, H] (row stride may be padded)."""
    assert q_fp4.ndim == 3 and q_fp4.shape[-1] * 2 == INDEX_HEAD_DIM, f"{q_fp4.shape=}"
    num_tokens, num_heads, _ = q_fp4.shape
    assert q_sf.shape == (num_tokens, num_heads), f"{q_sf.shape=}"
    q8, wq = _fp4_rows_to_fp8(q_fp4.contiguous(), q_sf.contiguous(), weights, num_heads)
    return PreparedQ(q8.view(num_tokens, num_heads, INDEX_HEAD_DIM), wq.view(num_tokens, num_heads))


def _alloc_logits(rows: int, width: int, device: torch.device) -> torch.Tensor:
    stride = triton.cdiv(max(width, 1), _ROW_ALIGN_ELEMS) * _ROW_ALIGN_ELEMS
    return torch.empty((rows, stride), dtype=torch.bfloat16, device=device)[:, :width]


@triton.jit
def _row_gather_kernel(
    q_ptr,  # [R, H, 128] e4m3
    wq_ptr,  # [R, H] fp32
    kv_ptr,  # PAGED: uint8 index-K cache base; else [N, 128] e4m3
    sk_ptr,  # flat: [N] fp32 key factors (unused when PAGED)
    bt_ptr,  # PAGED: [R, *] int32 block table
    cand_ptr,  # CAND: [R, *] int32 sorted sequence-local candidate blocks
    lens_ptr,  # [R] int32 visible kv count
    ks_ptr,  # flat: [R] int32 row offset into the flat kv
    out_ptr,  # [R, out_stride] bf16
    out_stride,
    stride_bt,
    stride_cand,
    page_stride,
    num_slots,
    max_cols,
    H: tl.constexpr,
    BN: tl.constexpr,
    PAGE: tl.constexpr,
    CBS: tl.constexpr,
    PAGED: tl.constexpr,
    CAND: tl.constexpr,
    SKIP_IDENTITY: tl.constexpr,
):
    row = tl.program_id(0)
    split = tl.program_id(1)
    nsplit = tl.num_programs(1)
    row64 = row.to(tl.int64)
    klen = tl.maximum(tl.load(lens_ptr + row), 0)
    if CAND:
        need_blocks = tl.cdiv(klen, CBS)
        ncols = tl.minimum(need_blocks, num_slots) * CBS
        if SKIP_IDENTITY:
            ncols = tl.where(need_blocks <= num_slots, 0, ncols)
    else:
        ncols = tl.minimum(klen, max_cols)
    ntiles = tl.cdiv(ncols, BN)
    if split < ntiles:
        d32 = tl.arange(0, 32)
        hh = tl.arange(0, H)
        q_cols = q_ptr + row64 * (H * 128) + hh[None, :] * 128 + d32[:, None]  # [32, H] = Q block^T
        q0 = tl.load(q_cols)
        q1 = tl.load(q_cols + 32)
        q2 = tl.load(q_cols + 64)
        q3 = tl.load(q_cols + 96)
        w = tl.load(wq_ptr + row64 * H + hh)
        if not PAGED:
            kv_base = tl.load(ks_ptr + row).to(tl.int64)
        for t in range(split, ntiles, nsplit):
            cols = t * BN + tl.arange(0, BN)
            cmask = cols < ncols
            if CAND:
                blk = tl.load(cand_ptr + row64 * stride_cand + cols // CBS, mask=cmask, other=-1)
                pos = blk * CBS + cols % CBS
                valid = cmask & (blk >= 0) & (pos < klen)
            else:
                pos = cols
                valid = cmask
            pos = tl.where(valid, pos, 0)
            if PAGED:
                page = tl.load(bt_ptr + row64 * stride_bt + pos // PAGE, mask=valid, other=0).to(tl.int64)
                off = (pos % PAGE).to(tl.int64)
                base = kv_ptr + page * page_stride
                sf = tl.load((base + PAGE * 64 + off * 4).to(tl.pointer_type(tl.int32)), mask=valid, other=0)
                e0, e1, e2, e3, _ = _sf_bytes(sf)
                b16 = tl.arange(0, 16)
                drow = base[:, None] + off[:, None] * 64 + b16[None, :]
                vm = valid[:, None]
                # unscaled e2m1 values in E4M3; each block's UE8M0 scale is applied to its exact 32-wide dot below
                k0 = _fp4_block_e4m3(tl.load(drow, mask=vm, other=0), BN)
                k1 = _fp4_block_e4m3(tl.load(drow + 16, mask=vm, other=0), BN)
                k2 = _fp4_block_e4m3(tl.load(drow + 32, mask=vm, other=0), BN)
                k3 = _fp4_block_e4m3(tl.load(drow + 48, mask=vm, other=0), BN)
                p0 = tl.dot(k0, q0, out_dtype=tl.float32) * _pow2_f32(e0 - 127)[:, None]
                p1 = tl.dot(k1, q1, out_dtype=tl.float32) * _pow2_f32(e1 - 127)[:, None]
                p2 = tl.dot(k2, q2, out_dtype=tl.float32) * _pow2_f32(e2 - 127)[:, None]
                p3 = tl.dot(k3, q3, out_dtype=tl.float32) * _pow2_f32(e3 - 127)[:, None]
                sk = 1.0
            else:
                kidx = kv_base + pos.to(tl.int64)
                k_rows = kv_ptr + kidx[:, None] * 128 + d32[None, :]
                vm = valid[:, None]
                k0 = tl.load(k_rows, mask=vm, other=0.0)
                k1 = tl.load(k_rows + 32, mask=vm, other=0.0)
                k2 = tl.load(k_rows + 64, mask=vm, other=0.0)
                k3 = tl.load(k_rows + 96, mask=vm, other=0.0)
                sk = tl.load(sk_ptr + kidx, mask=valid, other=0.0)
                p0 = tl.dot(k0, q0, out_dtype=tl.float32)  # [BN, H]
                p1 = tl.dot(k1, q1, out_dtype=tl.float32)
                p2 = tl.dot(k2, q2, out_dtype=tl.float32)
                p3 = tl.dot(k3, q3, out_dtype=tl.float32)
            s = (((p0 + p1) + p2) + p3).to(tl.bfloat16).to(tl.float32)
            s = tl.maximum(s, 0.0)
            s = (s * w[None, :]).to(tl.bfloat16).to(tl.float32)
            logit = (tl.sum(s, axis=1) * sk).to(tl.bfloat16)
            logit = tl.where(valid, logit, float("-inf"))
            tl.store(out_ptr + row64 * out_stride + cols, logit, mask=cmask)


def _grid_splits(rows: int, max_tiles: int, device: torch.device, programs_per_sm: int) -> int:
    """Programs per row so the grid covers the GPU; depends on shapes only (CUDA-graph safe)."""
    target = programs_per_sm * _num_sms(device.index if device.index is not None else torch.cuda.current_device())
    return max(1, min(max_tiles, triton.cdiv(target, max(rows, 1))))


def _launch_row_gather(
    q: PreparedQ,
    out: torch.Tensor,
    lens: torch.Tensor,
    *,
    kv: torch.Tensor,
    sk: torch.Tensor | None,
    block_table: torch.Tensor | None,
    ks: torch.Tensor | None,
    candidate_blocks: torch.Tensor | None,
    candidate_block_size: int,
    max_cols: int,
    skip_identity: bool,
) -> None:
    rows, num_heads, _ = q.q8.shape
    width = out.shape[1]
    if rows == 0 or width == 0:
        return
    paged = block_table is not None
    # H20 sweep (64 rows x 64K keys and 4K prefill rows x 16K candidates): paged rows are bound by the FP4
    # decode + MMA chain and want more programs in flight; flat rows load E4M3 keys and like deeper pipelining.
    block_n, num_stages, programs_per_sm = (128, 2, 16) if paged else (64, 3, 4)
    nsplit = _grid_splits(rows, triton.cdiv(width, block_n), out.device, programs_per_sm)
    page = kv.shape[1] if paged else 1
    dummy = lens
    _row_gather_kernel[(rows, nsplit)](
        q.q8,
        q.wq,
        kv,
        sk if sk is not None else dummy,
        block_table if paged else dummy,
        candidate_blocks if candidate_blocks is not None else dummy,
        lens,
        ks if ks is not None else dummy,
        out,
        out.stride(0),
        block_table.stride(0) if paged else 0,
        candidate_blocks.stride(0) if candidate_blocks is not None else 0,
        kv.stride(0) if paged else 0,
        candidate_blocks.shape[1] if candidate_blocks is not None else 0,
        max_cols,
        H=num_heads,
        BN=block_n,
        PAGE=page,
        CBS=max(candidate_block_size, 1),
        PAGED=paged,
        CAND=candidate_blocks is not None,
        SKIP_IDENTITY=skip_identity,
        num_warps=4,
        num_stages=num_stages,
    )


def _check_q(q: PreparedQ) -> None:
    assert q.q8.dtype == torch.float8_e4m3fn and q.q8.ndim == 3 and q.q8.shape[-1] == INDEX_HEAD_DIM and q.q8.is_contiguous()
    assert q.wq.dtype == torch.float32 and q.wq.shape == q.q8.shape[:2] and q.wq.is_contiguous()
    assert q.q8.shape[1] & (q.q8.shape[1] - 1) == 0 and q.q8.shape[1] >= 8, "head count must be a power of two >= 8"


def _check_i32(x: torch.Tensor, rows: int, name: str) -> None:
    assert x.dtype == torch.int32 and x.shape == (rows,) and x.is_contiguous(), f"{name}: {x.shape=} {x.dtype=}"


def _check_paged_cache(kv_cache: torch.Tensor) -> None:
    assert kv_cache.dtype == torch.uint8 and kv_cache.ndim == 4 and kv_cache.shape[2:] == (1, 68), f"{kv_cache.shape=}"
    assert kv_cache.stride(1) == 68 and kv_cache.stride(3) == 1 and kv_cache.stride(0) % 4 == 0, f"{kv_cache.stride()=}"


def paged_mqa_logits(
    q: PreparedQ, kv_cache: torch.Tensor, klen: torch.Tensor, block_tables: torch.Tensor, max_seqlen_k: int
) -> torch.Tensor:
    """Dense decode logits ``[B, max_seqlen_k]`` bf16 (SM90 ``deep_gemm.fp8_fp4_paged_mqa_logits``).

    ``kv_cache`` is the ``FP4_INDEX_KV_FORMAT`` page view ``[pages, page, 1, 68]`` (pages may be strided),
    ``block_tables`` [B, max_pages] int32 per query row, ``klen`` [B] int32.
    """
    _check_q(q)
    rows = q.q8.shape[0]
    _check_i32(klen, rows, "klen")
    _check_paged_cache(kv_cache)
    assert block_tables.dtype == torch.int32 and block_tables.shape[0] == rows and block_tables.stride(1) == 1
    out = _alloc_logits(rows, max_seqlen_k, q.q8.device)
    _launch_row_gather(
        q,
        out,
        klen,
        kv=kv_cache,
        sk=None,
        block_table=block_tables,
        ks=None,
        candidate_blocks=None,
        candidate_block_size=0,
        max_cols=max_seqlen_k,
        skip_identity=False,
    )
    return out


def paged_sparse_mqa_logits(
    q: PreparedQ,
    kv_cache: torch.Tensor,
    klen: torch.Tensor,
    block_tables: torch.Tensor,
    candidate_blocks: torch.Tensor,
    candidate_block_size: int,
) -> torch.Tensor:
    """Candidate decode logits ``[B, M * bs]`` bf16 (SM90 ``deep_gemm.fp8_fp4_paged_sparse_mqa_logits``)."""
    _check_q(q)
    rows = q.q8.shape[0]
    _check_i32(klen, rows, "klen")
    _check_paged_cache(kv_cache)
    assert block_tables.dtype == torch.int32 and block_tables.shape[0] == rows and block_tables.stride(1) == 1
    assert candidate_blocks.dtype == torch.int32 and candidate_blocks.shape[0] == rows and candidate_blocks.stride(1) == 1
    assert kv_cache.shape[1] % candidate_block_size == 0, "candidate blocks must not cross KV cache pages"
    width = candidate_blocks.shape[1] * candidate_block_size
    out = _alloc_logits(rows, width, q.q8.device)
    _launch_row_gather(
        q,
        out,
        klen,
        kv=kv_cache,
        sk=None,
        block_table=block_tables,
        ks=None,
        candidate_blocks=candidate_blocks,
        candidate_block_size=candidate_block_size,
        max_cols=width,
        skip_identity=False,
    )
    return out
