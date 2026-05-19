import triton
import triton.language as tl

@triton.autotune(configs=[
    triton.Config({'num_warps': 1}),
    triton.Config({'num_warps': 2}),
    triton.Config({'num_warps': 4}),
    triton.Config({'num_warps': 8}),
    triton.Config({'num_warps': 16}),
    triton.Config({'num_warps': 32}),
], key=['BT', 'BK', 'BV'])
@triton.jit
def chunk_delta_rule_fwd_kernel_h(
    k_ptr, v_ptr, d_ptr, v_new_ptr, initial_state_ptr, final_state_ptr,
    BT, BK, BV, NT,
    USE_INITIAL_STATE: tl.constexpr, STORE_FINAL_STATE: tl.constexpr
):
    i_bh = tl.program_id(0)
    i_k = tl.program_id(1)
    i_v = tl.program_id(2)

    # Calculate offsets and strides
    k_block_ptr = tl.make_block_ptr(k_ptr, shape=(BT, BK), strides=(BK, 1), offsets=(i_bh * BK, i_k))
    v_block_ptr = tl.make_block_ptr(v_ptr, shape=(BT, BV), strides=(BV, 1), offsets=(i_bh * BV, i_v))
    d_block_ptr = tl.make_block_ptr(d_ptr, shape=(BT, BK), strides=(BK, 1), offsets=(i_bh * BK, i_k))
    v_new_block_ptr = tl.make_block_ptr(v_new_ptr, shape=(BT, BV), strides=(BV, 1), offsets=(i_bh * BV, i_v))

    # Load initial state if required
    if USE_INITIAL_STATE:
        b_h = tl.load(initial_state_ptr + i_bh)

    # Iterate over time dimension
    for t in range(NT):
        k = tl.load(k_block_ptr + t * BK)
        v = tl.load(v_block_ptr + t * BV)
        d = tl.load(d_block_ptr + t * BK)

        # Compute update
        update = tl.dot(k, d)
        v_new = v + update

        # Store updated v
        tl.store(v_new_block_ptr + t * BV, v_new)

        # Update state
        if USE_INITIAL_STATE:
            b_h += update

    # Store final state if required
    if STORE_FINAL_STATE:
        tl.store(final_state_ptr + i_bh, b_h)


def chunk_fwd_h_fn(k, v, d, BT, BK, BV, NT, use_initial_state=False, store_final_state=False):
    # Calculate grid dimensions
    grid = (BT, BK, BV)

    # Initialize output tensors
    v_new = tl.zeros_like(v)
    h = None
    if use_initial_state:
        h = tl.zeros([BT])

    # Call the Triton kernel
    chunk_delta_rule_fwd_kernel_h[grid](
        k, v, d, v_new, h, h,
        BT, BK, BV, NT,
        USE_INITIAL_STATE=use_initial_state,
        STORE_FINAL_STATE=store_final_state
    )

    return v_new, h
