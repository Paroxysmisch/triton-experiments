import triton
import triton.language as tl

@triton.jit
def chunk_linear_attn_fwd_kernel_h(
    k_ptr, v_ptr, h_ptr,
    k_batch_stride, k_head_stride, k_time_stride, k_dim_stride,
    v_batch_stride, v_head_stride, v_time_stride, v_dim_stride,
    h_batch_stride, h_head_stride, h_time_stride, h_dim_stride,
    B, H, T, D,
    BLOCK_D: tl.constexpr,
    BLOCK_T: tl.constexpr
):
    b_idx = tl.program_id(0)
    h_idx = tl.program_id(1)
    t_idx = tl.program_id(2)
    d_idx = tl.program_id(3)

    k_block_ptr = tl.make_block_ptr(
        base=k_ptr,
        shape=(B, H, T, D),
        strides=(k_batch_stride, k_head_stride, k_time_stride, k_dim_stride),
        offsets=(b_idx, h_idx, t_idx, 0),
        block_shape=(1, 1, 1, BLOCK_D),
        order=(3, 0, 1, 2)
    )

    v_block_ptr = tl.make_block_ptr(
        base=v_ptr,
        shape=(B, H, T, D),
        strides=(v_batch_stride, v_head_stride, v_time_stride, v_dim_stride),
        offsets=(b_idx, h_idx, 0, 0),
        block_shape=(1, 1, BLOCK_T, BLOCK_D),
        order=(3, 0, 1, 2)
    )

    h_block_ptr = tl.make_block_ptr(
        base=h_ptr,
        shape=(B, H, T, D),
        strides=(h_batch_stride, h_head_stride, h_time_stride, h_dim_stride),
        offsets=(b_idx, h_idx, t_idx, 0),
        block_shape=(1, 1, 1, BLOCK_D),
        order=(3, 0, 1, 2)
    )

    k = tl.load(k_block_ptr)
    v = tl.load(v_block_ptr)

    h = tl.zeros((BLOCK_D,), dtype=tl.float32)

    for t in range(T):
        v_t = tl.load(v_block_ptr, mask=t < T, other=0.0)
        h += tl.dot(k, v_t)

    tl.store(h_block_ptr, h)

@triton.jit
def chunk_linear_attn_fwd_kernel_o(
    q_ptr, k_ptr, v_ptr, h_ptr, o_ptr,
    q_batch_stride, q_head_stride, q_time_stride, q_dim_stride,
    k_batch_stride, k_head_stride, k_time_stride, k_dim_stride,
    v_batch_stride, v_head_stride, v_time_stride, v_dim_stride,
    h_batch_stride, h_head_stride, h_time_stride, h_dim_stride,
    o_batch_stride, o_head_stride, o_time_stride, o_dim_stride,
    B, H, T, D,
    BLOCK_D: tl.constexpr,
    BLOCK_T: tl.constexpr
):
    b_idx = tl.program_id(0)
    h_idx = tl.program_id(1)
    t_idx = tl.program_id(2)
    d_idx = tl.program_id(3)

    q_block_ptr = tl.make_block_ptr(
        base=q_ptr,
        shape=(B, H, T, D),
        strides=(q_batch_stride, q_head_stride, q_time_stride, q_dim_stride),
        offsets=(b_idx, h_idx, t_idx, 0),
        block_shape=(1, 1, 1, BLOCK_D),
        order=(3, 0, 1, 2)
    )

    k_block_ptr = tl.make_block_ptr(
        base=k_ptr,
        shape=(B, H, T, D),
        strides=(k_batch_stride, k_head_stride, k_time_stride, k_dim_stride),
        offsets=(b_idx, h_idx, 0, 0),
        block_shape=(1, 1, BLOCK_T, BLOCK_D),
        order=(3, 0, 1, 2)
    )

    v_block_ptr = tl.make_block_ptr(
        base=v_ptr,
        shape=(B, H, T, D),
        strides=(v_batch_stride, v_head_stride, v_time_stride, v_dim_stride),
        offsets=(b_idx, h_idx, 0, 0),
        block_shape=(1, 1, BLOCK_T, BLOCK_D),
        order=(3, 0, 1, 2)
    )

    h_block_ptr = tl.make_block_ptr(
        base=h_ptr,
        shape=(B, H, T, D),
        strides=(h_batch_stride, h_head_stride, h_time_stride, h_dim_stride),
        offsets=(b_idx, h_idx, t_idx, 0),
        block_shape=(1, 1, 1, BLOCK_D),
        order=(3, 0, 1, 2)
    )

    o_block_ptr = tl.make_block_ptr(
        base=o_ptr,
        shape=(B, H, T, D),
        strides=(o_batch_stride, o_head_stride, o_time_stride, o_dim_stride),
        offsets=(b_idx, h_idx, t_idx, 0),
        block_shape=(1, 1, 1, BLOCK_D),
        order=(3, 0, 1, 2)
    )

    q = tl.load(q_block_ptr)
    k = tl.load(k_block_ptr)
    v = tl.load(v_block_ptr)
    h = tl.load(h_block_ptr)

    o = tl.zeros((BLOCK_D,), dtype=tl.float32)

    for t in range(T):
        k_t = tl.load(k_block_ptr, mask=t < T, other=0.0)
        v_t = tl.load(v_block_ptr, mask=t < T, other=0.0)
        o += tl.dot(q, v_t) * h

    tl.store(o_block_ptr, o)
