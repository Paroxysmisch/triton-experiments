import triton
import triton.language as tl

@triton.jit
def parallel_retention_fwd_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    q_stride0, q_stride1, q_stride2,
    k_stride0, k_stride1, k_stride2,
    v_stride0, v_stride1, v_stride2,
    o_stride0, o_stride1, o_stride2,
    n_heads, n_ctx, head_dim,
    decay_factor, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    # Compute the head index and the corresponding decay factor
    head_idx = pid % n_heads
    decay = decay_factor ** head_idx

    # Initialize pointers for the current block
    q_block_ptr = tl.make_block_ptr(
        base=q_ptr,
        shape=(n_ctx, n_heads, head_dim),
        strides=(q_stride0, q_stride1, q_stride2),
        offsets=(block_start, head_idx, 0),
        block_shape=(BLOCK_SIZE, 1, head_dim),
        order=(0, 2, 1)
    )
    k_block_ptr = tl.make_block_ptr(
        base=k_ptr,
        shape=(n_ctx, n_heads, head_dim),
        strides=(k_stride0, k_stride1, k_stride2),
        offsets=(block_start, head_idx, 0),
        block_shape=(BLOCK_SIZE, 1, head_dim),
        order=(0, 2, 1)
    )
    v_block_ptr = tl.make_block_ptr(
        base=v_ptr,
        shape=(n_ctx, n_heads, head_dim),
        strides=(v_stride0, v_stride1, v_stride2),
        offsets=(block_start, head_idx, 0),
        block_shape=(BLOCK_SIZE, 1, head_dim),
        order=(0, 2, 1)
    )
    o_block_ptr = tl.make_block_ptr(
        base=o_ptr,
        shape=(n_ctx, n_heads, head_dim),
        strides=(o_stride0, o_stride1, o_stride2),
        offsets=(block_start, head_idx, 0),
        block_shape=(BLOCK_SIZE, 1, head_dim),
        order=(0, 2, 1)
    )

    # Load q, k, v blocks
    q = tl.load(q_block_ptr)
    k = tl.load(k_block_ptr)
    v = tl.load(v_block_ptr)

    # Compute attention scores
    scores = tl.dot(q, k, trans_b=True) * decay
    scores = tl.softmax(scores, axis=1)

    # Compute output
    o = tl.dot(scores, v)

    # Store the output
    tl.store(o_block_ptr, o)

@triton.jit
def _parallel_retention_bwd_dq(
    q_ptr, k_ptr, v_ptr, do_ptr, dq_ptr,
    q_stride0, q_stride1, q_stride2,
    k_stride0, k_stride1, k_stride2,
    v_stride0, v_stride1, v_stride2,
    do_stride0, do_stride1, do_stride2,
    dq_stride0, dq_stride1, dq_stride2,
    n_heads, n_ctx, head_dim,
    decay_factor, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    # Compute the head index and the corresponding decay factor
    head_idx = pid % n_heads
    decay = decay_factor ** head_idx

    # Initialize pointers for the current block
    q_block_ptr = tl.make_block_ptr(
        base=q_ptr,
        shape=(n_ctx, n_heads, head_dim),
        strides=(q_stride0, q_stride1, q_stride2),
        offsets=(block_start, head_idx, 0),
        block_shape=(BLOCK_SIZE, 1, head_dim),
        order=(0, 2, 1)
    )
    k_block_ptr = tl.make_block_ptr(
        base=k_ptr,
        shape=(n_ctx, n_heads, head_dim),
        strides=(k_stride0, k_stride1, k_stride2),
        offsets=(block_start, head_idx, 0),
        block_shape=(BLOCK_SIZE, 1, head_dim),
        order=(0, 2, 1)
    )
    v_block_ptr = tl.make_block_ptr(
        base=v_ptr,
        shape=(n_ctx, n_heads, head_dim),
        strides=(v_stride0, v_stride1, v_stride2),
        offsets=(block_start, head_idx, 0),
        block_shape=(BLOCK_SIZE, 1, head_dim),
        order=(0, 2, 1)
    )
    do_block_ptr = tl.make_block_ptr(
        base=do_ptr,
        shape=(n_ctx, n_heads, head_dim),
        strides=(do_stride0, do_stride1, do_stride2),
        offsets=(block_start, head_idx, 0),
        block_shape=(BLOCK_SIZE, 1, head_dim),
        order=(0, 2, 1)
    )
    dq_block_ptr = tl.make_block_ptr(
        base=dq_ptr,
        shape=(n_ctx, n_heads, head_dim),
        strides=(dq_stride0, dq_stride1, dq_stride2),
        offsets=(block_start, head_idx, 0),
        block_shape=(BLOCK_SIZE, 1, head_dim),
        order=(0, 2, 1)
    )

    # Load q, k, v, do blocks
    q = tl.load(q_block_ptr)
    k = tl.load(k_block_ptr)
    v = tl.load(v_block_ptr)
    do = tl.load(do_block_ptr)

    # Compute attention scores
    scores = tl.dot(q, k, trans_b=True) * decay
    scores = tl.softmax(scores, axis=1)

    # Compute dq
    dv = tl.dot(scores, do)
    dq = tl.dot(dv, k, trans_b=True) * decay

    # Store the gradient
    tl.store(dq_block_ptr, dq)
