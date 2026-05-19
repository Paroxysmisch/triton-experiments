import triton
import triton.language as tl

@triton.jit
def _triton_rope(q_ptr, k_ptr, cos, sin, BACKWARD_PASS, batch_size, num_heads, head_dim, stride_q, stride_k, stride_cos_sin, **meta):
    pid = tl.program_id(axis=0)
    
    # Calculate batch, head, and position indices
    batch_id = pid // (num_heads * head_dim)
    head_id = (pid // head_dim) % num_heads
    pos_id = pid % head_dim
    
    # Offset pointers for the current batch and head
    q_offset = batch_id * stride_q + head_id * head_dim + pos_id
    k_offset = batch_id * stride_k + head_id * head_dim + pos_id
    cos_sin_offset = pos_id
    
    # Load the current values of q, k, cos, and sin
    q_val = tl.load(q_ptr + q_offset)
    k_val = tl.load(k_ptr + k_offset)
    cos_val = tl.load(cos + cos_sin_offset)
    sin_val = tl.load(sin + cos_sin_offset)
    
    # Apply the rotary embedding
    if BACKWARD_PASS:
        # Apply inverse rotation
        q_new = q_val * cos_val + k_val * sin_val
        k_new = k_val * cos_val - q_val * sin_val
    else:
        # Apply forward rotation
        q_new = q_val * cos_val - k_val * sin_val
        k_new = k_val * cos_val + q_val * sin_val
    
    # Store the results back
    tl.store(q_ptr + q_offset, q_new)
    tl.store(k_ptr + k_offset, k_new)

def rope_backward(dq, dk, q, k, cos, sin, batch_size, num_heads, head_dim, BACKWARD_PASS):
    # Transpose and align the dimensions for efficient memory access
    q_aligned = tl.make_aligned(q, 128)
    k_aligned = tl.make_aligned(k, 128)
    dq_aligned = tl.make_aligned(dq, 128)
    dk_aligned = tl.make_aligned(dk, 128)
    
    # Calculate strides
    stride_q = q_aligned.stride(0)
    stride_k = k_aligned.stride(0)
    stride_cos_sin = cos.stride(0)
    
    # Launch the Triton kernel
    grid = (batch_size * num_heads * head_dim,)
    _triton_rope[grid](
        q_aligned, k_aligned, cos, sin, BACKWARD_PASS,
        batch_size, num_heads, head_dim,
        stride_q, stride_k, stride_cos_sin
    )

# Example usage:
# Assume q, k, cos, sin are properly initialized GPU tensors
# batch_size, num_heads, head_dim are set accordingly
# dq, dk are the gradient placeholders for q and k
# rope_backward(dq, dk, q, k, cos, sin, batch_size, num_heads, head_dim, BACKWARD_PASS=True)
