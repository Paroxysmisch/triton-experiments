import torch
import triton
import triton.language as tl

@triton.jit
def _triton_rope(
    q_ptr, k_ptr, cos_ptr, sin_ptr, 
    q_out_ptr, k_out_ptr,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_km, stride_kk,
    stride_cos, stride_sin,
    BLOCK_SIZE: tl.constexpr
):
    # Define block indices
    pid = tl.program_id(0)
    
    # Compute offsets for each dimension
    q_offset = pid * BLOCK_SIZE
    k_offset = pid * BLOCK_SIZE
    
    # Load q and k blocks
    q = tl.load(q_ptr + q_offset, mask=q_offset < stride_qm)
    k = tl.load(k_ptr + k_offset, mask=k_offset < stride_km)
    
    # Load cos and sin blocks
    cos = tl.load(cos_ptr + q_offset, mask=q_offset < stride_cos)
    sin = tl.load(sin_ptr + q_offset, mask=q_offset < stride_sin)
    
    # Apply the ROPE transformation
    q_cos = q * cos - k * sin
    k_sin = q * sin + k * cos
    
    # Store the results
    tl.store(q_out_ptr + q_offset, q_cos, mask=q_offset < stride_qm)
    tl.store(k_out_ptr + k_offset, k_sin, mask=k_offset < stride_km)

def rope_forward(q, k, cos, sin):
    # Ensure inputs are on the correct device
    assert q.is_cuda and k.is_cuda and cos.is_cuda and sin.is_cuda, "All inputs must be CUDA tensors"
    
    # Get shapes and ensure they are compatible
    assert q.shape == k.shape, "q and k must have the same shape"
    assert cos.shape == sin.shape, "cos and sin must have the same shape"
    
    # Prepare output tensors
    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)
    
    # Get the dimensions and calculate strides
    B, H, M, K = q.shape
    stride_qz, stride_qh, stride_qm, stride_qk = q.stride()
    stride_kz, stride_kh, stride_km, stride_kk = k.stride()
    stride_cos = cos.stride(0)
    stride_sin = sin.stride(0)
    
    # Define block size
    BLOCK_SIZE = 128  # This can be tuned for performance
    
    # Launch the Triton kernel
    num_blocks = (M * K + BLOCK_SIZE - 1) // BLOCK_SIZE
    _triton_rope[num_blocks](
        q, k, cos, sin, q_out, k_out,
        stride_qz, stride_qh, stride_qm, stride_qk,
        stride_kz, stride_kh, stride_km, stride_kk,
        stride_cos, stride_sin,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return q_out, k_out, cos, sin

# Example usage:
# q, k, cos, sin = torch.randn(...), torch.randn(...), torch.randn(...), torch.randn(...)
# q_transformed, k_transformed, cos, sin = rope_forward(q, k, cos, sin)
