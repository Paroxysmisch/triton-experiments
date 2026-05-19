import triton
import triton.language as tl

# Constants
BK = 32  # Block size for keys
BV = 32  # Block size for values
scale = 0.01  # Scale factor for queries

# Triton kernel for the forward pass
@triton.jit
def fused_recurrent_retention_fwd_kernel(
    q, k, v, o, h, do, i_h, initial_state, use_initial_state, store_final_state,
    T, B, H, BK, BV, scale, BLOCK_SIZE: tl.constexpr
):
    # Thread indices
    pid = tl.program_id(axis=0)
    i = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    i_h = tl.arange(0, H)

    # Initialize h if using initial state
    if use_initial_state:
        h = initial_state

    # Iterate over the temporal dimension
    for t in range(T):
        # Load query, key, and value blocks
        q_block = tl.load(q + i * T * H * BK + t * H * BK, mask=i < B)
        k_block = tl.load(k + i * T * H * BK + t * H * BK, mask=i < B)
        v_block = tl.load(v + i * T * H * BV + t * H * BV, mask=i < B)

        # Scale query
        q_block *= scale

        # Compute attention scores
        scores = tl.dot(q_block, k_block.T, allow_tf32=True)

        # Apply softmax to scores
        exp_scores = tl.exp(scores - tl.max(scores, axis=1, keepdims=True))
        softmax_scores = exp_scores / tl.sum(exp_scores, axis=1, keepdims=True)

        # Compute weighted sum of values
        o_block = tl.dot(softmax_scores, v_block, allow_tf32=True)

        # Update accumulator h
        h = h * (1 - i_h) + o_block * i_h

        # Store output block
        tl.store(o + i * T * H + t * H, o_block, mask=i < B)

    # Store final state if required
    if store_final_state:
        tl.store(h, h, mask=i < B)

# Triton kernel for the backward pass
@triton.jit
def fused_recurrent_retention_bwd_kernel(
    q, k, v, do, dh, dq, dk, dv, i_h, initial_state, use_initial_state, store_final_state,
    T, B, H, BK, BV, scale, BLOCK_SIZE: tl.constexpr
):
    # Thread indices
    pid = tl.program_id(axis=0)
    i = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    i_h = tl.arange(0, H)

    # Initialize dh if using initial state
    if use_initial_state:
        dh = initial_state

    # Iterate over the temporal dimension in reverse
    for t in range(T - 1, -1, -1):
        # Load query, key, value blocks
        q_block = tl.load(q + i * T * H * BK + t * H * BK, mask=i < B)
        k_block = tl.load(k + i * T * H * BK + t * H * BK, mask=i < B)
        v_block = tl.load(v + i * T * H * BV + t * H * BV, mask=i < B)

        # Load output and gradient blocks
        o_block = tl.load(do + i * T * H + t * H, mask=i < B)
        dh_block = tl.load(dh + i * T * H + t * H, mask=i < B)

        # Scale query
        q_block *= scale

        # Compute attention scores
        scores = tl.dot(q_block, k_block.T, allow_tf32=True)

        # Apply softmax to scores
        exp_scores = tl.exp(scores - tl.max(scores, axis=1, keepdims=True))
        softmax_scores = exp_scores / tl.sum(exp_scores, axis=1, keepdims=True)

        # Compute gradients
        grad_o_block = dh_block
        grad_v_block = tl.dot(softmax_scores.T, grad_o_block, allow_tf32=True)
        grad_k_block = tl.dot(softmax_scores, grad_o_block * q_block.T, allow_tf32=True)
        grad_q_block = tl.dot(grad_o_block, v_block.T, allow_tf32=True) * softmax_scores

        # Update gradients
        tl.store(dq + i * T * H * BK + t * H * BK, grad_q_block, mask=i < B)
        tl.store(dk + i * T * H * BK + t * H * BK, grad_k_block, mask=i < B)
        tl.store(dv + i * T * H * BV + t * H * BV, grad_v_block, mask=i < B)

        # Update dh for previous timestep
        dh_block = tl.dot(grad_o_block, k_block.T, allow_tf32=True)
        tl.store(dh + i * T * H + (t - 1) * H, dh_block, mask=i < B)

    # Store final state if required
    if store_final_state:
        tl.store(dh, dh, mask=i < B)

# Wrapper function to handle the computation
def fused_recurrent_retention(
    q, k, v, o, h, do, i_h, initial_state=None, use_initial_state=False, store_final_state=False
):
    T, B, H = q.shape[0], q.shape[1], q.shape[2]
    BLOCK_SIZE = 32  # Example block size, can be adjusted

    # Launch forward kernel
    fused_recurrent_retention_fwd_kernel[T, BLOCK_SIZE](q, k, v, o, h, do, i_h, initial_state, use_initial_state, store_final_state, T, B, H, BK, BV, scale, BLOCK_SIZE)

    # Launch backward kernel
    fused_recurrent_retention_bwd_kernel[T, BLOCK_SIZE](q, k, v, do, h, None, None, None, i_h, initial_state, use_initial_state, store_final_state, T, B, H, BK, BV, scale, BLOCK_SIZE)

# Example usage
# q = ...  # Query tensor of shape (T, B, H, BK)
# k = ...  # Key tensor of shape (T, B, H, BK)
# v = ...  # Value tensor of shape (T, B, H, BV)
# o = ...  # Output tensor of shape (T, B, H)
# h = ...  # Hidden state tensor of shape (B, H)
# do = ...  # Gradient of output tensor of shape (T, B, H)
# i_h = ...  # Head indices tensor of shape (H)
# initial_state = ...  # Optional initial state tensor of shape (B, H)

# fused_recurrent_retention(q, k, v, o, h, do, i_h, initial_state, use_initial_state=True, store_final_state=True)
