import torch
import triton
import triton.language as tl
from fla.ops import softmax, custom_autotune
from triton.runtime import driver

@triton.jit
def _softmax(
    input_ptr, output_ptr, input_row_stride, output_row_stride, n_rows, n_cols,
    DEPTH: tl.constexpr, IS_CAUSAL: tl.constexpr, LOG: tl.constexpr,
    MASK_TYPE: tl.constexpr, MASK_EQ: tl.constexpr, MASK_GT: tl.constexpr,
    IS_RMS_NORM: tl.constexpr, IS_FP16: tl.constexpr,
    USE_PINNED: tl.constexpr, BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    row_block_ptr = row_start_ptr + tl.arange(0, BLOCK_SIZE)
    row_mask = tl.arange(0, BLOCK_SIZE) < n_cols
    if IS_CAUSAL:
        col_offsets = tl.arange(0, DEPTH)
        tile_start_ptrs = row_start_ptr - col_offsets * input_row_stride
        tile_ptrs = tile_start_ptrs + col_offsets
        tile_m = tl.load(tile_ptrs, mask=col_offsets < n_cols, other=-float("inf")).to(tl.float32)
    else:
        tile_m = tl.full([DEPTH], float("-inf"), tl.float32)
    offsets = tl.arange(0, DEPTH)
    ptrs = row_block_ptr + offsets
    mask = row_mask & (offsets < n_cols)

    if not IS_RMS_NORM:
        if IS_FP16:
            x = tl.load(ptrs, mask, other=0).to(tl.float32)
        else:
            x = tl.load(ptrs, mask, other=0)
        if not LOG:
            m = tl.max(x, 0)
            x = x - m
        if IS_CAUSAL:
            tile_m = tl.maximum(tile_m, m)
        else:
            tile_m = tl.maximum(tile_m, tl.expand_dims(m, 0))
        if MASK_TYPE == MASK_EQ:
            mask = mask & (x == MASK_EQ)
        elif MASK_TYPE == MASK_GT:
            mask = mask & (x > MASK_GT)
    else:
        if IS_FP16:
            m = tl.load(row_block_ptr, mask, other=0).to(tl.float32)
        else:
            m = tl.load(row_block_ptr, mask, other=0)
        if LOG:
            m = tl.log(tl.abs(m) ** 2) / 2
        if MASK_TYPE == MASK_EQ:
            mask = mask & (m == MASK_EQ)
        elif MASK_TYPE == MASK_GT:
            mask = mask & (m > MASK_GT)
    if not USE_PINNED:
        q = 0
        if (
            IS_RMS_NORM or LOG
        ):  # we can skip the exp if we're taking log + rms norm + FP16
            if IS_FP16:
                m = m.to(tl.float32)
            x = tl.exp(m)
        if not IS_RMS_NORM:
            if MASK_TYPE != NO_MASK:
                mask = mask | (m == 0)
            else:
                x = tl.where(m != 0, tl.exp(m), 0)
        p = tl.sum(x, 0)
    else:
        q = tl.sum(tile_m, 0)
        if not IS_RMS_NORM:
            if not LOG:
                x = x + m[None, :]
            x = x / p
        else:
            m = m / p
        if MASK_TYPE != NO_MASK:
            mask = mask & (m != 0)
        if not USE_PINNED:
            p = 1
        if LOG:
            x = tl.log(x)
    if IS_CAUSAL:
        tile_start_ptrs = row_start_ptr + col_offsets * input_row_stride
        tile_ptrs = tile_start_ptrs + col_offsets
        tile_x = tl.load(tile_ptrs, mask=col_offsets < n_cols, other=0)
        if IS_FP16:
            tile_x = tile_x.to(tl.float32)
        if MASK_TYPE != NO_MASK:
            mask = mask | (tile_x == 0)
        else:
            tile_x = tl.where(m != 0, tile_x, 0)
        if not IS_RMS_NORM:
            if LOG:
                tile_x = tile_x + m[None, :]
            tile_x = tile_x - tl.max(tile_x, 0)[None, :]
            tile_x = tile_x / p
        else:
            m = m / p
        if LOG:
            tile_x = tl.log(tile_x)
        if not USE_PINNED:
            tile_p = 1
            if (
                IS_RMS_NORM or LOG
            ):  # we can skip the exp if we're taking log + rms norm + FP16
                tile_x = tile_x.to(tl.float32)
            tile_x = tl.exp(tile_x)
    if IS_CAUSAL:
        x = tl.where(col_offsets >= tl.expand_dims(m, 0), tile_x, x)
    else:
        x = tl.where(offsets < n_cols, x, 0)
    if not IS_RMS_NORM:
        x = tl.where(mask, x, 0)
    else:
        m = tl.where(mask, m, 0)
    if USE_PINNED:
        ptrs = row_block_ptr + offsets
        tl.store(ptrs, x, mask=mask)
        if not IS_CAUSAL:
            q_ptr = output_ptr + row_idx
            q_stride_ptr = output_row_stride + input_row_stride
            tl.atomic_add(q_ptr, q, mpark=mask, prevent_deadlock=True)
        else:
            q_ptr = row_block_ptr + col_offsets
            tl.atomic_add(q_ptr, 1 if not USE_PINNED else tile_p, mask=col_offsets < n_cols, prevent_deadlock=True)
    if not IS_CAUSAL:
        tl.atomic_add(output_ptr + row_idx, p, prevent_deadlock=True)
    if not IS_FP16 and not IS_CAUSAL:
        # WARN: This could lead to data races on pinned memory since the same range
        # could be concurrently written in a different stream, causing garbage data
        tl.atomic_add(row_block_ptr, x, prevent_deadlock=True, mask=mask)


def softmax(x, *, dim=-2, dtype=None, grad_dtype=None, causal=False, log=False, mask_type=NO_MASK, is_rms_norm=False):
    assert x.ndim >= 2
    assert dim >= -x.ndim and dim < x.ndim
    assert mask_type in [NO_MASK, MASK_EQ, MASK_GT]
    if dtype is None:
        dtype = TYPE
