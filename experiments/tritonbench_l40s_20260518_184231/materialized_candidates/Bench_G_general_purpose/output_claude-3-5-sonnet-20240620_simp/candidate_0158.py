import torch
import triton
import triton.language as tl

@triton.jit
def rmsnorm_triton(
    x_ptr,          # pointer to input tensor [B, M, N]
    rms_w_ptr,      # pointer to weight vector [N]
    output_ptr,     # pointer to output tensor [B, M, N]
    batch_stride,   # stride for batch dimension
    m_stride,       # stride for M dimension
    n_stride,       # stride for N dimension
    B,              # batch size
    M,              # sequence length
    N,              # hidden dimension
    BLOCK_N_SIZE:   # block size for N dimension
    tl.constexpr,
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate batch and m indices
    batch_id = pid // M
    m_id = pid % M
    
    # Compute pointer offsets
    batch_offset = batch_id * batch_stride
    m_offset = m_id * m_stride
    
    # Create offsets for the N dimension
    offs_n = tl.arange(0, BLOCK_N_SIZE)
    mask_n = offs_n < N
    
    # Load input slice
    x_ptrs = x_ptr + batch_offset + m_offset + offs_n * n_stride
    x = tl.load(x_ptrs, mask=mask_n, other=0.0)
    
    # Compute RMS norm
    x2 = x * x
    rms_sum = tl.sum(x2, axis=0) / N
    rms = tl.sqrt(rms_sum + 1e-6)  # Add epsilon for numerical stability
    
    # Normalize
    x_norm = x / rms
    
    # Load and apply weight
    w_ptrs = rms_w_ptr + offs_n
    w = tl.load(w_ptrs, mask=mask_n, other=1.0)
    output = x_norm * w
    
    # Store result
    out_ptrs = output_ptr + batch_offset + m_offset + offs_n * n_stride
    tl.store(out_ptrs, output, mask=mask_n)

def rmsnorm_triton_wrapper(x: torch.Tensor, rms_weight: torch.Tensor) -> torch.Tensor:
    """
    Wrapper function for RMSNorm Triton kernel
    Args:
        x: Input tensor of shape [B, M, N]
        rms_weight: Weight vector of shape [N]
    Returns:
        Normalized tensor of shape [B, M, N]
    """
    assert x.dim() == 3, "Input tensor must be 3D"
    assert rms_weight.dim() == 1, "Weight tensor must be 1D"
    
    B, M, N = x.shape
    output = torch.empty_like(x)
    
    # Ensure tensors are contiguous and in correct format
    x = x.contiguous()
    rms_weight = rms_weight.contiguous()
    
    # Calculate strides
    batch_stride = x.stride(0)
    m_stride = x.stride(1)
    n_stride = x.stride(2)
    
    # Choose BLOCK_N_SIZE as power of 2 close to N
    BLOCK_N_SIZE = triton.next_power_of_2(N)
    
    # Launch kernel
    grid = (B * M,)  # One thread block per (batch, M) pair
    rmsnorm_triton[grid](
        x_ptr=x,
        rms_w_ptr=rms_weight,
        output_ptr=output,
        batch_stride=batch_stride,
        m_stride=m_stride,
        n_stride=n_stride,
        B=B,
        M=M,
        N=N,
        BLOCK_N_SIZE=BLOCK_N_SIZE,
    )
    
    return output
