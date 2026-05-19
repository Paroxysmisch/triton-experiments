import torch
import triton
import triton.language as tl
from torch.cuda.amp import custom_bwd, custom_fwd


@triton.jit
def chunk_retention_fwd_kernel_h(
    q_ptr, k_ptr, v_ptr, h_ptr, initial_state_ptr, final_state_ptr,
    s_q_h, s_q_t, s_q_d, s_k_h, s_k_t, s_k_d, s_v_h, s_v_t, s_v_d, s_h_h, s_h_t, s_h_d,
    B, H, T, scale,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    DK: tl.constexpr, DV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    boundary_check: tl.constexpr
):
    # Program ID determines which block of data we operate on
    block_n = tl.program_id(0)  # which chunk along the T dimension
    block_bh = tl.program_id(1)  # which batch-head
    b_idx = block_bh // H
    h_idx = block_bh % H

    # Offsets for reading/writing from input / to state
    off_t = block_n * BT
    # Indices within our block
    t_range = tl.arange(0, BT)

    # Pointers
    q_offset = b_idx * s_q_h + h_idx * s_q_t + off_t * s_q_d
    k_offset = b_idx * s_k_h + h_idx * s_k_t + off_t * s_k_d
    v_offset = b_idx * s_v_h + h_idx * s_v_t + off_t * s_v_d
    h_offset = b_idx * s_h_h + h_idx * s_h_t + off_t * s_h_d

    # Load initial hidden state if needed
    hidden = tl.zeros((BK, BV), dtype=tl.float32)
    if USE_INITIAL_STATE and (block_n == 0):
        init_offset = b_idx * DK * DV * H + h_idx * DK * DV
        p_init = tl.make_block_ptr(
            base=initial_state_ptr + init_offset,
            shape=(DK, DV),
            strides=(DV, 1),
            offsets=(0, 0),
            block_shape=(BK, BV),
            order=(1, 0)
        )
        hidden = tl.load(p_init, boundary_check=(1, 1))

    # Make block pointers to Q, K, V, and output H
    p_q = tl.make_block_ptr(
        base=q_ptr + q_offset,
        shape=(T, DK),
        strides=(s_q_t, s_q_d),
        offsets=(0, 0),
        block_shape=(BT, BK),
        order=(1, 0)
    )
    p_k = tl.make_block_ptr(
        base=k_ptr + k_offset,
        shape=(T, DK),
        strides=(s_k_t, s_k_d),
        offsets=(0, 0),
        block_shape=(BT, BK),
        order=(1, 0)
    )
    p_v = tl.make_block_ptr(
        base=v_ptr + v_offset,
        shape=(T, DV),
        strides=(s_v_t, s_v_d),
        offsets=(0, 0),
        block_shape=(BT, BV),
        order=(1, 0)
    )
    p_h = tl.make_block_ptr(
        base=h_ptr + h_offset,
        shape=(T, DV),
        strides=(s_h_t, s_h_d),
        offsets=(0, 0),
        block_shape=(BT, BV),
        order=(1, 0)
    )

    # Load Q, K, V
    q_block = tl.load(p_q, boundary_check=(1 if boundary_check else 0, 1 if boundary_check else 0))
    k_block = tl.load(p_k, boundary_check=(1 if boundary_check else 0, 1 if boundary_check else 0))
    v_block = tl.load(p_v, boundary_check=(1 if boundary_check else 0, 1 if boundary_check else 0))

    # Scale Q
    q_block = q_block * scale
    # Compute retention hidden state updates
    # This is a simplified conceptual example
    # Weighted sum for hidden state
    update = tl.dot(k_block.to(q_block.dtype), v_block, allow_tf32=False).to(hidden.dtype)
    hidden = hidden + update

    # Output
    out = tl.dot(q_block, hidden, allow_tf32=False)
    tl.store(p_h, out, boundary_check=(1 if boundary_check else 0, 1 if boundary_check else 0))

    # Optionally store final state
    if STORE_FINAL_STATE and (block_n == (T // BT) - 1):
        final_offset = b_idx * DK * DV * H + h_idx * DK * DV
        p_final = tl.make_block_ptr(
            base=final_state_ptr + final_offset,
            shape=(DK, DV),
            strides=(DV, 1),
            offsets=(0, 0),
            block_shape=(BK, BV),
            order=(1, 0)
        )
        tl.store(p_final, hidden, boundary_check=(1, 1))


@triton.jit
def chunk_retention_fwd_kernel_o(
    q_ptr, k_ptr, v_ptr, h_ptr, o_ptr,
    s_q_h, s_q_t, s_q_d,
    s_k_h, s_k_t, s_k_d,
    s_v_h, s_v_t, s_v_d,
    s_h_h, s_h_t, s_h_d,
    s_o_h, s_o_t, s_o_d,
    B, H, T, scale,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    DK: tl.constexpr, DV: tl.constexpr,
    boundary_check: tl.constexpr
):
    # Program IDs
    block_n = tl.program_id(0)
    block_bh = tl.program_id(1)
    b_idx = block_bh // H
    h_idx = block_bh % H
    off_t = block_n * BT
    t_range = tl.arange(0, BT)

    # Offsets
    q_offset = b_idx * s_q_h + h_idx * s_q_t + off_t * s_q_d
    k_offset = b_idx * s_k_h + h_idx * s_k_t + off_t * s_k_d
    v_offset = b_idx * s_v_h + h_idx * s_v_t + off_t * s_v_d
    h_offset = b_idx * s_h_h + h_idx * s_h_t + off_t * s_h_d
    o_offset = b_idx * s_o_h + h_idx * s_o_t + off_t * s_o_d

    # Make block pointers
    p_q = tl.make_block_ptr(
        base=q_ptr + q_offset,
        shape=(T, DK),
        strides=(s_q_t, s_q_d),
        offsets=(0, 0),
        block_shape=(BT, BK),
        order=(1, 0)
    )
    p_k = tl.make_block_ptr(
        base=k_ptr + k_offset,
        shape=(T, DK),
        strides=(s_k_t, s_k_d),
        offsets=(0, 0),
        block_shape=(BT, BK),
        order=(1, 0)
    )
    p_v = tl.make_block_ptr(
        base=v_ptr + v_offset,
        shape=(T, DV),
        strides=(s_v_t, s_v_d),
        offsets=(0, 0),
        block_shape=(BT, BV),
        order=(1, 0)
    )
    p_h = tl.make_block_ptr(
        base=h_ptr + h_offset,
        shape=(T, DV),
        strides=(s_h_t, s_h_d),
        offsets=(0, 0),
        block_shape=(BT, BV),
        order=(1, 0)
    )
    p_o = tl.make_block_ptr(
        base=o_ptr + o_offset,
        shape=(T, DV),
        strides=(s_o_t, s_o_d),
        offsets=(0, 0),
        block_shape=(BT, BV),
        order=(1, 0)
    )

    # Load
    q_block = tl.load(p_q, boundary_check=(1 if boundary_check else 0, 1 if boundary_check else 0))
    k_block = tl.load(p_k, boundary_check=(1 if boundary_check else 0, 1 if boundary_check else 0))
    v_block = tl.load(p_v, boundary_check=(1 if boundary_check else 0, 1 if boundary_check else 0))
    h_block = tl.load(p_h, boundary_check=(1 if boundary_check else 0, 1 if boundary_check else 0))

    # Scale Q
    q_block *= scale
    # Compute output
    attn_weights = tl.dot(q_block, k_block, allow_tf32=False)
    attn_out = tl.dot(attn_weights, v_block, allow_tf32=False)
    # Combine with hidden
    out = attn
