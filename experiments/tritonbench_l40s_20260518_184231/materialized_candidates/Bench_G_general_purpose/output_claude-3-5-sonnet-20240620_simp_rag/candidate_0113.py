import triton
import triton.language as tl
import torch

@triton.jit
def rmsnorm_triton(
    x_ptr: tl.pointer_type,
    rms_weights_ptr: tl.pointer_type,
    out_ptr: tl.pointer_type,
    stride_b: tl.int32,
    stride_m: tl.int32,
    N: tl.int32,
    M: tl.int32,
    K: tl.int32,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate batch and sequence indices
    batch_idx = pid // M
    seq_idx = pid % M
    
    # Calculate start pointer for this instance
    x_start_ptr = x_ptr + batch_idx * stride_b + seq_idx * stride_m
    out_start_ptr = out_ptr + batch_idx * stride_b + seq_idx * stride_m
    
    # Load K elements for this instance
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < K
    x = tl.load(x_start_ptr + offs, mask=mask, other=0.0)
    
    # Calculate RMS
    square_sum = tl.sum(x * x, axis=0) / K
    rms = tl.sqrt(square_sum + 1e-6)  # Add epsilon for numerical stability
    
    # Normalize
    x_normalized = x / rms
    
    # Apply weights
    weights = tl.load(rms_weights_ptr + offs, mask=mask, other=1.0)
    output = x_normalized * weights
    
    # Store result
    tl.store(out_start_ptr + offs, output, mask=mask)

def rmsnorm_wrapper(x: torch.Tensor, rms_weights: torch.Tensor) -> torch.Tensor:
    """
    Wrapper function for RMS normalization kernel
    
    Args:
        x: Input tensor of shape (B, M, K)
        rms_weights: Weights tensor of shape (K,)
    
    Returns:
        Normalized tensor of shape (B, M, K)
    """
    assert x.is_cuda and rms_weights.is_cuda, "Input tensors must be on GPU"
    assert len(x.shape) == 3, "Input tensor must be 3D (B, M, K)"
    assert len(rms_weights.shape) == 1, "Weights tensor must be 1D (K,)"
    assert x.shape[2] == rms_weights.shape[0], "Last dimension must match weights dimension"
    
    B, M, K = x.shape
    
    # Create output tensor
    output = torch.empty_like(x)
    
    # Calculate strides
    stride_b = M * K
    stride_m = K
    
    # Calculate block size (next power of 2)
    BLOCK_SIZE = triton.next_power_of_2(K)
    
    # Launch kernel
    grid = (B * M,)
    rmsnorm_triton[grid](
        x_ptr=x,
        rms_weights_ptr=rms_weights,
        out_ptr=output,
        stride_b=stride_b,
        stride_m=stride_m,
        N=B,
        M=M,
        K=K,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
    )
    
    return output
