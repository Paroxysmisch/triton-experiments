triton
@triton.jit
def _fwd_kernel(
    Q, K, V, o, max_out, denom, sm_scale, Lk, num_heads, sequence_length, batch_size, IS_CAUSAL,
    BLOCK_M: int, BLOCK_N: int, BLOCK_K: int, num_warps: int
):
    # Define block indices
    pid = triton.program_id(0)
    grid_m = sequence_length // BLOCK_M
    grid_n = num_heads * batch_size
    grid_k = Lk // BLOCK_K

    # Compute the indices for Q, K, V, o, max_out, and denom
    m = pid // (grid_n * grid_k) * BLOCK_M
    n = (pid // grid_k) % grid_n
    k = pid % grid_k * BLOCK_K

    # Initialize pointers
    q_ptr = Q + m * Q.stride(0) + n * Q.stride(1) + k * Q.stride(2)
    k_ptr = K + m * K.stride(0) + n * K.stride(1) + k * K.stride(2)
    v_ptr = V + m * V.stride(0) + n * V.stride(1) + k * V.stride(2)
    o_ptr = o + m * o.stride(0) + n * o.stride(1) + k * o.stride(2)
    max_ptr = max_out + m * max_out.stride(0) + n * max_out.stride(1) + k * max_out.stride(2)
    denom_ptr = denom + m * denom.stride(0) + n * denom.stride(1) + k * denom.stride(2)

    # Load Q, K, V segments
    q = triton.load(q_ptr, mask=(m < sequence_length) & (n < num_heads * batch_size) & (k < Lk))
    k = triton.load(k_ptr, mask=(m < sequence_length) & (n < num_heads * batch_size) & (k < Lk))
    v = triton.load(v_ptr, mask=(m < sequence_length) & (n < num_heads * batch_size) & (k < Lk))

    # Initialize max_out and denom
    max_val = -float('inf')
    denom_val = 0.0

    # Compute dot products and softmax
    for i in range(BLOCK_K):
        if k + i < Lk:
            qk = q * k[i]
            max_val = triton.maximum(max_val, qk)
            denom_val += exp2(qk - max_val)

    # Normalize and store results
    if denom_val > 0:
        scaled_val = v * exp2(qk - max_val - log2(denom_val))
        triton.atomic.add(o_ptr, scaled_val, mask=(m < sequence_length) & (n < num_heads * batch_size) & (k < Lk))
        triton.atomic.max(max_ptr, max_val, mask=(m < sequence_length) & (n < num_heads * batch_size) & (k < Lk))
        triton.atomic.add(denom_ptr, denom_val, mask=(m < sequence_length) & (n < num_heads * batch_size) & (k < Lk))

    # Apply causal masking if enabled
    if IS_CAUSAL:
        causal_mask = (m + k < sequence_length)
        triton.atomic.add(o_ptr, o_ptr * causal_mask, mask=(m < sequence_length) & (n < num_heads * batch_size) & (k < Lk))
        triton.atomic.max(max_ptr, max_val * causal_mask, mask=(m < sequence_length) & (n < num_heads * batch_size) & (k < Lk))
        triton.atomic.add(denom_ptr, denom_val * causal_mask, mask=(m < sequence_length) & (n < num_heads * batch_size) & (k < Lk))
