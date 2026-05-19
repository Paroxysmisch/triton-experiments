import triton
import triton.language as tl

@triton.jit
def chunk_delta_rule_fwd_kernel_h(
    k_ptr, v_ptr, d_ptr, v_new_ptr, h_ptr, initial_state_ptr, final_state_ptr,
    K_BLOCK_SIZE: tl.constexpr, V_BLOCK_SIZE: tl.constexpr, D_BLOCK_SIZE: tl.constexpr,
    H_BLOCK_SIZE: tl.constexpr, NT: tl.constexpr, B: tl.constexpr, H: tl.constexpr,
    S: tl.constexpr, C: tl.constexpr, store_initial_state: tl.constexpr, store_final_state: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_blocks = (B * H * S) // (K_BLOCK_SIZE * V_BLOCK_SIZE * D_BLOCK_SIZE * H_BLOCK_SIZE)
    if pid >= num_blocks:
        return

    # Compute block indices
    b_idx = pid // (H * S)
    h_idx = (pid % (H * S)) // S
    s_idx = (pid % (H * S)) % S

    # Initialize block pointers
    k_block_ptr = tl.make_block_ptr(
        base=k_ptr, shape=(B, H, S, C), strides=(H * S * C, S * C, C, 1),
        offsets=(b_idx * K_BLOCK_SIZE, h_idx * H_BLOCK_SIZE, s_idx * H_BLOCK_SIZE, 0),
        block_shape=(K_BLOCK_SIZE, H_BLOCK_SIZE, H_BLOCK_SIZE, C),
        order=(0, 1, 2, 3)
    )
    v_block_ptr = tl.make_block_ptr(
        base=v_ptr, shape=(B, H, S, C), strides=(H * S * C, S * C, C, 1),
        offsets=(b_idx * V_BLOCK_SIZE, h_idx * H_BLOCK_SIZE, s_idx * H_BLOCK_SIZE, 0),
        block_shape=(V_BLOCK_SIZE, H_BLOCK_SIZE, H_BLOCK_SIZE, C),
        order=(0, 1, 2, 3)
    )
    d_block_ptr = tl.make_block_ptr(
        base=d_ptr, shape=(B, H, S, C), strides=(H * S * C, S * C, C, 1),
        offsets=(b_idx * D_BLOCK_SIZE, h_idx * H_BLOCK_SIZE, s_idx * H_BLOCK_SIZE, 0),
        block_shape=(D_BLOCK_SIZE, H_BLOCK_SIZE, H_BLOCK_SIZE, C),
        order=(0, 1, 2, 3)
    )
    v_new_block_ptr = tl.make_block_ptr(
        base=v_new_ptr, shape=(B, H, S, C), strides=(H * S * C, S * C, C, 1),
        offsets=(b_idx * V_BLOCK_SIZE, h_idx * H_BLOCK_SIZE, s_idx * H_BLOCK_SIZE, 0),
        block_shape=(V_BLOCK_SIZE, H_BLOCK_SIZE, H_BLOCK_SIZE, C),
        order=(0, 1, 2, 3)
    )
    h_block_ptr = tl.make_block_ptr(
        base=h_ptr, shape=(B, H, S, C), strides=(H * S * C, S * C, C, 1),
        offsets=(b_idx * H_BLOCK_SIZE, h_idx * H_BLOCK_SIZE, s_idx * H_BLOCK_SIZE, 0),
        block_shape=(H_BLOCK_SIZE, H_BLOCK_SIZE, H_BLOCK_SIZE, C),
        order=(0, 1, 2, 3)
    )
    initial_state_block_ptr = tl.make_block_ptr(
        base=initial_state_ptr, shape=(B, H, S, C), strides=(H * S * C, S * C, C, 1),
        offsets=(b_idx * H_BLOCK_SIZE, h_idx * H_BLOCK_SIZE, s_idx * H_BLOCK_SIZE, 0),
        block_shape=(H_BLOCK_SIZE, H_BLOCK_SIZE, H_BLOCK_SIZE, C),
        order=(0, 1, 2, 3)
    )
    final_state_block_ptr = tl.make_block_ptr(
        base=final_state_ptr, shape=(B, H, S, C), strides=(H * S * C, S * C, C, 1),
        offsets=(b_idx * H_BLOCK_SIZE, h_idx * H_BLOCK_SIZE, s_idx * H_BLOCK_SIZE, 0),
        block_shape=(H_BLOCK_SIZE, H_BLOCK_SIZE, H_BLOCK_SIZE, C),
        order=(0, 1, 2, 3)
    )

    # Load initial state if needed
    if store_initial_state:
        initial_state = tl.load(initial_state_block_ptr)
    else:
        initial_state = tl.zeros((H_BLOCK_SIZE, H_BLOCK_SIZE, H_BLOCK_SIZE, C), dtype=tl.float32)

    # Main loop over time dimension
    for t in range(NT):
        k = tl.load(k_block_ptr)
        v = tl.load(v_block_ptr)
        d = tl.load(d_block_ptr)

        # Compute dot products and cumulative sums
        h = tl.dot(k, v) + initial_state
        v_new = h * d

        # Store results
        tl.store(h_block_ptr, h)
        tl.store(v_new_block_ptr, v_new)

        # Update initial state for next iteration
        initial_state = h

    # Store final state if needed
    if store_final_state:
        tl.store(final_state_block_ptr, initial_state)

import torch
import triton
import triton.language as tl

def chunk_fwd_h_fn(k, v, d, v_new, h, initial_state, final_state, NT, B, H, S, C, store_initial_state, store_final_state):
    # Initialize output tensors
    v_new.zero_()
    h.zero_()
    if store_initial_state:
        initial_state.zero_()
    if store_final_state:
        final_state.zero_()

    # Calculate grid and block sizes
    K_BLOCK_SIZE = 16
    V_BLOCK_SIZE = 16
    D_BLOCK_SIZE = 16
    H_BLOCK_SIZE = 16
    grid = (B * H * S) // (K_BLOCK_SIZE * V_BLOCK_SIZE * D_BLOCK_SIZE * H_BLOCK_SIZE)

    # Invoke the Triton kernel
    chunk_delta_rule_fwd_kernel_h[grid](
        k, v, d, v_new, h, initial_state, final_state,
        K_BLOCK_SIZE, V_BLOCK_SIZE, D_BLOCK_SIZE, H_BLOCK_SIZE, NT, B, H, S, C,
        store_initial_state, store_final_state
    )

    return v_new, h, initial_state, final_state
