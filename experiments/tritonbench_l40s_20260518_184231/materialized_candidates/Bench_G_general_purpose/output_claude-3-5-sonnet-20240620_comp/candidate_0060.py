import torch
import triton
import triton.language as tl
import math

@triton.jit
def _triton_rope(
    q_ptr, k_ptr, cos_ptr, sin_ptr, 
    stride_q_b, stride_q_h, stride_q_s, stride_q_d,
    stride_k_b, stride_k_h, stride_k_s, stride_k_d,
    stride_cos_s, stride_cos_d,
    stride_sin_s, stride_sin_d,
    batch_size, n_heads, seq_len, dim_head,
    BLOCK_SIZE: tl.constexpr,
    BACKWARD_PASS: tl.constexpr,
):
    # Program ID for parallel execution
    pid = tl.program_id(0)
    
    # Calculate indices for batch, head, and sequence
    batch_id = pid // (n_heads * seq_len)
    head_id = (pid % (n_heads * seq_len)) // seq_len
    seq_id = pid % seq_len

    # Calculate base pointers for this instance
    q_base = q_ptr + batch_id * stride_q_b + head_id * stride_q_h + seq_id * stride_q_s
    k_base = k_ptr + batch_id * stride_k_b + head_id * stride_k_h + seq_id * stride_k_s
    cos_base = cos_ptr + seq_id * stride_cos_s
    sin_base = sin_ptr + seq_id * stride_sin_s

    # Half dimension for RoPE (split into two parts)
    dim_half = dim_head // 2

    # Load offsets for the dimension
    offs_d = tl.arange(0, BLOCK_SIZE)
    mask_d = offs_d < dim_half

    # Process in blocks of size BLOCK_SIZE
    for d in range(0, dim_half, BLOCK_SIZE):
        # Load q and k values
        q_real = tl.load(q_base + d * 2, mask=mask_d)
        q_imag = tl.load(q_base + d * 2 + 1, mask=mask_d)
        k_real = tl.load(k_base + d * 2, mask=mask_d)
        k_imag = tl.load(k_base + d * 2 + 1, mask=mask_d)

        # Load cos and sin values
        cos = tl.load(cos_base + d, mask=mask_d)
        sin = tl.load(sin_base + d, mask=mask_d)

        if not BACKWARD_PASS:
            # Forward pass rotation
            q_out_real = q_real * cos - q_imag * sin
            q_out_imag = q_real * sin + q_imag * cos
            k_out_real = k_real * cos - k_imag * sin
            k_out_imag = k_real * sin + k_imag * cos
        else:
            # Backward pass rotation (inverse)
            q_out_real = q_real * cos + q_imag * sin
            q_out_imag = -q_real * sin + q_imag * cos
            k_out_real = k_real * cos + k_imag * sin
            k_out_imag = -k_real * sin + k_imag * cos

        # Store results
        tl.store(q_base + d * 2, q_out_real, mask=mask_d)
        tl.store(q_base + d * 2 + 1, q_out_imag, mask=mask_d)
        tl.store(k_base + d * 2, k_out_real, mask=mask_d)
        tl.store(k_base + d * 2 + 1, k_out_imag, mask=mask_d)

def rope_forward(q, k, cos, sin):
    # Get dimensions
    batch_size, n_heads, seq_len, dim_head = q.shape
    
    # Ensure dimensions are power of 2 for efficiency
    block_size = min(triton.next_power_of_2(dim_head // 2), 128)
    
    # Launch kernel
    grid = (batch_size * n_heads * seq_len,)
    _triton_rope[grid](
        q, k, cos, sin,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        cos.stride(0), cos.stride(1),
        sin.stride(0), sin.stride(1),
        batch_size, n_heads, seq_len, dim_head,
        BLOCK_SIZE=block_size,
        BACKWARD_PASS=False,
    )
    return q, k

def rope_backward(dq, dk, cos, sin):
    # Transpose gradients for backward pass
    dq = dq.contiguous()
    dk = dk.contiguous()
    
    # Get dimensions
    batch_size, n_heads, seq_len, dim_head = dq.shape
    
    # Ensure dimensions are power of 2 for efficiency
    block_size = min(triton.next_power_of_2(dim_head // 2), 128)
    
    # Launch kernel with backward configuration
    grid = (batch_size * n_heads * seq_len,)
    _triton_rope[grid](
        dq, dk, cos, sin,
        dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
        dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
        cos.stride(0), cos.stride(1),
        sin.stride(0), sin.stride(1),
        batch_size, n_heads, seq_len, dim_head,
        BLOCK_SIZE=block_size,
        BACKWARD_PASS=True,
    )
    return dq, dk
