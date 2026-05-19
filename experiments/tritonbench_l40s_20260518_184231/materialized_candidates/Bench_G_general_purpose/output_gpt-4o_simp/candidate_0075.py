import triton
import triton.language as tl

# Triton kernel for forward pass operation
@triton.jit
def chunk_simple_gla_fwd_kernel_o(q_ptr, k_ptr, v_ptr, h_ptr, o_ptr,
                                  BT, BK, BV,
                                  stride_qb, stride_qk,
                                  stride_kb, stride_kk,
                                  stride_vb, stride_vk,
                                  stride_hb, stride_hk,
                                  stride_ob, stride_oo,
                                  BLOCK_SIZE: tl.constexpr):
    # Program ID for parallel execution
    pid = tl.program_id(0)
    
    # Compute block start indices
    block_start_b = pid // BT
    block_start_k = (pid % BT) * BLOCK_SIZE

    # Pointers to the current block
    q_block_ptr = q_ptr + block_start_b * stride_qb + block_start_k * stride_qk
    k_block_ptr = k_ptr + block_start_b * stride_kb + block_start_k * stride_kk
    v_block_ptr = v_ptr + block_start_b * stride_vb + block_start_k * stride_vk
    h_block_ptr = h_ptr + block_start_b * stride_hb + block_start_k * stride_hk
    o_block_ptr = o_ptr + block_start_b * stride_ob + block_start_k * stride_oo

    # Load q, k, v, h into shared memory
    q = tl.load(q_block_ptr + tl.arange(0, BLOCK_SIZE)[:, None] * stride_qk)
    k = tl.load(k_block_ptr + tl.arange(0, BLOCK_SIZE)[None, :] * stride_kk)
    v = tl.load(v_block_ptr + tl.arange(0, BLOCK_SIZE)[:, None] * stride_vk)
    h = tl.load(h_block_ptr + tl.arange(0, BLOCK_SIZE)[:, None] * stride_hk)

    # Compute dot product q * k^T
    qk = tl.dot(q, k)

    # Apply exponential transformation and scaling
    qk_exp = tl.exp(qk)
    qk_scaled = qk_exp * h

    # Compute the output
    o = tl.dot(qk_scaled, v)

    # Store the result
    tl.store(o_block_ptr + tl.arange(0, BLOCK_SIZE)[:, None] * stride_oo, o)


# Wrapper function to prepare parameters and launch the kernel
def chunk_fwd_o_fn(q, k, v, h, o, BT, BK, BV):
    # Get the shape and strides of the input tensors
    q_shape, q_strides = q.shape, q.strides
    k_shape, k_strides = k.shape, k.strides
    v_shape, v_strides = v.shape, v.strides
    h_shape, h_strides = h.shape, h.strides
    o_shape, o_strides = o.shape, o.strides

    # Define grid size
    grid = (BT * (q_shape[1] // BK),)

    # Launch the kernel
    chunk_simple_gla_fwd_kernel_o[grid](
        q, k, v, h, o,
        BT, BK, BV,
        q_strides[0], q_strides[1],
        k_strides[0], k_strides[1],
        v_strides[0], v_strides[1],
        h_strides[0], h_strides[1],
        o_strides[0], o_strides[1],
        BLOCK_SIZE=BK
    )
