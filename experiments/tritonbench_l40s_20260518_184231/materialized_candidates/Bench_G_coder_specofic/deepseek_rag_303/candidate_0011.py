import torch
import triton
import triton.language as tl
import numpy as np

@triton.jit
def rms_matmul_rbe(
    x_ptr,
    w_ptr,
    w_row_stride,
    w_batch_stride,
    emb_ptr,
    emb_row_stride,
    rms_w_ptr,
    rms_x_ptr,
    output_ptr,
    output_row_stride,
    output_batch_stride,
    THETA: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BATCH_SIZE: tl.constexpr,
    SRC_LEN: tl.constexpr,
    HEADS: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    NUM_ROTARY_EMBEDDINGS_M: tl.constexpr,
    CACHE_KEY_M: tl.constexpr,
    CACHE_KEY_N: tl.constexpr
):
    # Triton kernel for matrix multiplication with weight and optional rotary embeddings.
    pid = tl.program_id(axis=0)
    batch_idx = pid // HEADS
    head_idx = pid % HEADS
    col_offsets = tl.arange(0, BLOCK_SIZE_N)
    m_row_start_idx = (batch_idx * BLOCK_SIZE_M)
    mask = (m_row_start_idx + col_offsets) < (BATCH_SIZE * HEADS)
    # initial value, mask will be updated later in the loop
    output = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)
    w_block_ptr = tl.make_block_ptr(
        base=w_ptr,
        shape=(HEAD_SIZE, HEADS * NUM_ROTARY_EMBEDDINGS_M),
        strides=(w_row_stride, w_batch_stride),
        offsets=(head_idx * HEAD_SIZE, batch_idx * NUM_ROTARY_EMBEDDINGS_M),
        block_shape=(BLOCK_SIZE_N, 1),
        order=(1, 0),
    )
    rms_w_block_ptr = tl.make_block_ptr(
        base=rms_w_ptr,
        shape=(HEADS,),
        strides=(1,),
        offsets=(head_idx,),
        block_shape=(1,),
        order=(0,)
    )

    for k in range(0, tl.cdiv(SRC_LEN, HEAD_SIZE)):
        if THETA != 0:
            emb_scale = k * (HEAD_SIZE // 2)
            row_offsets = tl.arange(0, BLOCK_SIZE_M)
            emb_row_start_idx = ((batch_idx * HEADS * tl.cdiv(SRC_LEN, HEAD_SIZE))
                                 + ((k * BLOCK_SIZE_M) + m_row_start_idx))
            emb_mask = (emb_row_start_idx + row_offsets) < (BATCH_SIZE * HEADS * tl.cdiv(SRC_LEN, HEAD_SIZE))
            emb_ptrs = emb_ptr + (emb_row_start_idx * emb_row_stride) + emb_scale * emb_row_stride
            cos = tl.cos(THETA * emb_scale)
            sin = tl.sin(THETA * emb_scale)
            # NOTE: Isaac can optimize this memory access by keeping it in L2
            # Overparametrization is more efficient if inputs are too huge to fit in L2.
            # emb = tl.load(emb_ptrs, mask=emb_mask, other=0.0)
            emb_real = tl.load(emb_ptrs, mask=emb_mask, other=0.0)
            emb_imag = tl.load(emb_ptrs + emb_row_stride, mask=emb_mask, other=0.0)

            # Emb [m, n]
            emb_real_br = tl.broadcast_to(emb_real[:, None], [BLOCK_SIZE_M, BLOCK_SIZE_N])
            emb_imag_br = tl.broadcast_to(emb_imag[:, None], [BLOCK_SIZE_M, BLOCK_SIZE_N])
            x_rot_real = emb_real_br * cos - emb_imag_br * sin
            x_rot_imag = emb_real_br * sin + emb_imag_br * cos
        else:
            x_rot_real = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            x_rot_imag = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        rms_x_block_ptr = tl.make_block_ptr(
            base=rms_x_ptr,
            shape=(BATCH_SIZE * HEADS, HEAD_SIZE),
            strides=(1, NUM_ROTARY_EMBEDDINGS_M),
            offsets=(batch_idx * HEADS + head_idx, k * HEAD_SIZE),
            block_shape=(BLOCK_SIZE_M, HEAD_SIZE),
            order=(1, 0),
        )
        x = tl.load(rms_x_block_ptr, boundary_check=(0, 1), padding_option='zero')

        x_real = x_rot_real + x
        x_imag = x_rot_imag + x
        x_norm_squared_real = tl.sum(x_real * x_real, axis=1)
        x_norm_squared_imag = tl.sum(x_imag * x_imag, axis=1)
        x_norm_real = tl.sqrt(x_norm_squared_real)
        w = tl.load(w_block_ptr, boundary_check=(0, 1))
        rms_w = tl.load(rms_w_block_ptr, boundary_check=(0,))

        w_real = w
        w_imag = tl.zeros_like(w)
        output += (x_real * w_real - x_imag * w_imag) * rms_w
        tl.debug_barrier()

        w_block_ptr = tl.advance(w_block_ptr, (0, -NUM_ROTARY_EMBEDDINGS_M))
        w_block_ptr = tl.advance(w_block_ptr, (1,HEAD_SIZE))
        mask = tl.advance_to(mask, w_block_ptr)
        output += (x_imag * w_real + x_real * w_imag) * rms_w
        tl.debug_barrier()
        w_block_ptr = tl.advance(w_block_ptr, (-1 * HEAD_SIZE, 0))

        rms_w = tl.load(rms_w_block_ptr, boundary_check=(0,))
        output *= rms_w
        rms_w_block_ptr = tl.advance(rms_w_block_ptr, (-1,))
    output_block_ptr = tl.make_block_ptr(
        base=output_ptr,
        shape=(BATCH_SIZE * HEADS, tl.cdiv(SRC_LEN, HEAD_SIZE) * BLOCK_SIZE_N),
        strides=(output_batch_stride, output_row_stride),
        offsets=(batch_idx * HEADS + head_idx, m_row_start_idx),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        order=(1, 0)
    )

    tl.store(output_block_ptr, output, boundary_check=(0, 1))


def
