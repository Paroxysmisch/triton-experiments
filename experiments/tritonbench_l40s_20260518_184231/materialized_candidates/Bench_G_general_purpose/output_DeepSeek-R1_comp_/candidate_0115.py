import torch
import triton
import triton.language as tl

@triton.jit
def rmsnorm_triton(
    # Pointers to tensors
    x_ptr, rms_w_ptr, out_ptr,
    # Size of the K dimension
    N_SIZE, eps,
    # Strides for x
    x_batch_stride, x_m_stride, x_k_stride,
    # Strides for rms_weight
    rms_w_k_stride,
    # Strides for output
    out_batch_stride, out_m_stride, out_k_stride,
    # Block size for K dimension
    BLOCK_N_SIZE: tl.constexpr,
):
    # Get the current batch and M index
    batch_idx = tl.program_id(0)
    m_idx = tl.program_id(1)
    
    # Compute the base pointers for the current batch and M in x and output
    x_start_ptr = x_ptr + batch_idx * x_batch_stride + m_idx * x_m_stride
    out_start_ptr = out_ptr + batch_idx * out_batch_stride + m_idx * out_m_stride
    
    sum_squares = 0.0
    
    # Loop over K to compute sum of squares
    for k_offset in range(0, N_SIZE, BLOCK_N_SIZE):
        k_indices = k_offset + tl.arange(0, BLOCK_N_SIZE)
        mask = k_indices < N_SIZE
        
        x_ptrs = x_start_ptr + k_indices * x_k_stride
        x = tl.load(x_ptrs, mask=mask, other=0.0)
        x_square = x * x
        sum_squares += tl.sum(x_square, axis=0)
    
    # Compute RMS value
    mean = sum_squares / N_SIZE
    rms_val = 1.0 / tl.sqrt(mean + eps)
    
    # Loop over K again to compute and store normalized output
    for k_offset in range(0, N_SIZE, BLOCK_N_SIZE):
        k_indices = k_offset + tl.arange(0, BLOCK_N_SIZE)
        mask = k_indices < N_SIZE
        
        x_ptrs = x_start_ptr + k_indices * x_k_stride
        w_ptrs = rms_w_ptr + k_indices * rms_w_k_stride
        out_ptrs = out_start_ptr + k_indices * out_k_stride
        
        x = tl.load(x_ptrs, mask=mask, other=0.0)
        w = tl.load(w_ptrs, mask=mask, other=0.0)
        out = x * rms_val * w
        tl.store(out_ptrs, out, mask=mask)

def rmsnorm_wrapper(x: torch.Tensor, rms_weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    # Ensure tensors are on GPU and contiguous
    assert x.is_cuda and rms_weight.is_cuda
    assert x.is_contiguous() and rms_weight.is_contiguous()
    
    batch_size, M_size, K_size = x.shape
    output = torch.empty_like(x)
    
    # Define grid dimensions (parallel over batch and M)
    grid = (batch_size, M_size)
    
    # Set block size for K dimension (tune based on hardware)
    BLOCK_N_SIZE = 128
    
    # Launch kernel with appropriate parameters
    rmsnorm_triton[grid](
        x, rms_weight, output,
        K_size, eps,
        x.stride(0), x.stride(1), x.stride(2),
        rms_weight.stride(0),
        output.stride(0), output.stride(1), output.stride(2),
        BLOCK_N_SIZE=BLOCK_N_SIZE,
        num_warps=BLOCK_N_SIZE // 32  # 4 warps for 128 threads
    )
    
    return output

# Example Usage
if __name__ == "__main__":
    batch, M, K = 4, 256, 512
    x = torch.randn(batch, M, K, device='cuda', dtype=torch.float16)
    rms_weight = torch.randn(K, device='cuda', dtype=torch.float16)
    
    output = rmsnorm_wrapper(x, rms_weight)
    print("Output shape:", output.shape)  # Should be [4, 256, 512]
