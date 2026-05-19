import triton
import triton.language as tl
import torch

@triton.jit
def rmsnorm_triton(
    x_ptr: tl.pointer_type,
    rms_w_ptr: tl.pointer_type,
    out_ptr: tl.pointer_type,
    stride_b: tl.int32,
    stride_m: tl.int32,
    N_SIZE: tl.int32,
    eps: tl.float32,
    BLOCK_N_SIZE: tl.constexpr,
):
    # Get program ID for batch and M dimensions
    batch_id = tl.program_id(0)
    m_id = tl.program_id(1)
    
    # Calculate offset for current batch and M position
    base_offset = batch_id * stride_b + m_id * stride_m
    
    # Create offsets for the K dimension
    offs = tl.arange(0, BLOCK_N_SIZE)
    mask = offs < N_SIZE
    
    # Load input values
    x = tl.load(x_ptr + base_offset + offs, mask=mask, other=0.0)
    
    # Compute RMS
    square_sum = tl.sum(x * x, axis=0) / N_SIZE
    rms = tl.sqrt(square_sum + eps)
    
    # Normalize and apply weights
    rms_weights = tl.load(rms_w_ptr + offs, mask=mask, other=1.0)
    output = (x / rms) * rms_weights
    
    # Store result
    tl.store(out_ptr + base_offset + offs, output, mask=mask)

def rmsnorm_wrapper(x: torch.Tensor, rms_weights: torch.Tensor, eps: float = 1e-6):
    """
    Wrapper function for RMSNorm Triton kernel
    Args:
        x: Input tensor of shape [batch, M, K]
        rms_weights: Weights tensor of shape [K]
        eps: Small constant for numerical stability
    """
    assert x.is_cuda and rms_weights.is_cuda, "Input tensors must be on GPU"
    assert len(x.shape) == 3, "Input tensor must be 3D [batch, M, K]"
    
    batch, M, K = x.shape
    
    # Ensure input is contiguous
    x = x.contiguous()
    rms_weights = rms_weights.contiguous()
    
    # Compute strides
    stride_b = M * K
    stride_m = K
    
    # Determine block size (power of 2)
    BLOCK_N_SIZE = triton.next_power_of_2(K)
    
    # Allocate output tensor
    output = torch.empty_like(x)
    
    # Launch kernel
    grid = (batch, M)
    rmsnorm_triton[grid](
        x_ptr=x,
        rms_w_ptr=rms_weights,
        out_ptr=output,
        stride_b=stride_b,
        stride_m=stride_m,
        N_SIZE=K,
        eps=eps,
        num_warps=4,
        BLOCK_N_SIZE=BLOCK_N_SIZE,
    )
    
    return output
