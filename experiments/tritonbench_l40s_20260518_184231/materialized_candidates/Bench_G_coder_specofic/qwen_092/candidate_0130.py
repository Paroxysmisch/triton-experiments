triton
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_K': 1}, num_stages=1, num_warps=1),
        triton.Config({'BLOCK_SIZE_K': 2}, num_stages=1, num_warps=2),
        triton.Config({'BLOCK_SIZE_K': 4}, num_stages=1, num_warps=4),
        triton.Config({'BLOCK_SIZE_K': 8}, num_stages=1, num_warps=8),
        triton.Config({'BLOCK_SIZE_K': 16}, num_stages=1, num_warps=16),
        triton.Config({'BLOCK_SIZE_K': 32}, num_stages=1, num_warps=32),
    ],
    key=['BT', 'BK', 'BV']
)
@triton.jit
def chunk_delta_rule_fwd_kernel_h(
    k: ptr,  # [BT, BK, NT, K]
    v: ptr,  # [BT, BK, NT, V]
    d: ptr,  # [BT, BK, NT, V]
    h: ptr,  # [BT, BK, NT, V] (optional)
    v_new: ptr,  # [BT, BK, NT, V]
    initial_state: ptr,  # [BT, BK, V] (optional)
    final_state: ptr,  # [BT, BK, V] (optional)
    BT: int32,
    BK: int32,
    NT: int32,
    K: int32,
    V: int32,
    BLOCK_SIZE_K: int32,
    USE_INITIAL_STATE: int32,
    STORE_FINAL_STATE: int32,
    stride_k: int32,
    stride_v: int32,
    stride_d: int32,
    stride_h: int32,
    stride_v_new: int32,
    stride_initial_state: int32,
    stride_final_state: int32
):
    i_k = tl.program_id(axis=0)
    i_v = tl.program_id(axis=1)
    i_bh = tl.program_id(axis=2)

    k_ptr = tl.make_block_ptr(
        base=k,
        index=[i_k, i_v, 0, 0],
        shape=[BT, BK, NT, K],
        strides=[stride_k, stride_v * NT * K, stride_k, 1],
        block_shape=[1, BLOCK_SIZE_K, 1, 1],
        order=[0, 1, 2, 3]
    )

    v_ptr = tl.make_block_ptr(
        base=v,
        index=[i_k, i_v, 0, 0],
        shape=[BT, BK, NT, V],
        strides=[stride_v, stride_v * NT * V, stride_v, 1],
        block_shape=[1, BLOCK_SIZE_K, 1, 1],
        order=[0, 1, 2, 3]
    )

    d_ptr = tl.make_block_ptr(
        base=d,
        index=[i_k, i_v, 0, 0],
        shape=[BT, BK, NT, V],
        strides=[stride_d, stride_d * NT * V, stride_d, 1],
        block_shape=[1, BLOCK_SIZE_K, 1, 1],
        order=[0, 1, 2, 3]
    )

    h_ptr = tl.make_block_ptr(
        base=h,
        index=[i_k, i_v, 0, 0],
        shape=[BT, BK, NT, V],
        strides=[stride_h, stride_h * NT * V, stride_h, 1],
        block_shape=[1, BLOCK_SIZE_K, 1, 1],
        order=[0, 1, 2, 3]
    )

    v_new_ptr = tl.make_block_ptr(
        base=v_new,
        index=[i_k, i_v, 0, 0],
        shape=[BT, BK, NT, V],
        strides=[stride_v_new, stride_v_new * NT * V, stride_v_new, 1],
        block_shape=[1, BLOCK_SIZE_K, 1, 1],
        order=[0, 1, 2, 3]
    )

    initial_state_ptr = tl.make_block_ptr(
        base=initial_state,
        index=[i_k, i_v, 0],
        shape=[BT, BK, V],
        strides=[stride_initial_state, stride_initial_state * V, 1],
        block_shape=[1, BLOCK_SIZE_K, 1],
        order=[0, 1, 2]
    )

    final_state_ptr = tl.make_block_ptr(
        base=final_state,
        index=[i_k, i_v, 0],
        shape=[BT, BK, V],
        strides=[stride_final_state, stride_final_state * V, 1],
        block_shape=[1, BLOCK_SIZE_K, 1],
        order=[0, 1, 2]
    )

    b_h = tl.zeros([BLOCK_SIZE_K, V], dtype=tl.float32)
    if USE_INITIAL_STATE:
        b_h += tl.load(initial_state_ptr)

    for n in range(NT):
        k_n = tl.load(k_ptr + n * stride_k)
        d_n = tl.load(d_ptr + n * stride_d)
        v_n = tl.load(v_ptr + n * stride_v)

        b_h = b_h + tl.dot(k_n, d_n, allow_tf32=False)

        tl.store(v_new_ptr + n * stride_v_new, v_n + b_h)

    if STORE_FINAL_STATE:
        tl.store(final_state_ptr, b_h)
