import triton
import triton.language as tl
import torch

@triton.jit
def rmsnorm_triton(
    x_ptr: tl.pointer_type,
    rms_w_ptr: tl.pointer_type,
    output_ptr: tl.pointer_type,
    stride_x: tl.int32,
    stride_w: tl.int32,
    stride_out: tl.int32,
    N_SIZE: tl.int32,
    eps: tl.float32,
    BLOCK_N_SIZE: tl.constexpr,
):
    # Get the program ID for batch and M dimension
    pid_batch = tl.program_id(0)
    pid_m = tl.program_id(1)
    
    # Compute the starting offset for this instance
    offset_x = pid_batch * stride_x + pid_m * N_SIZE
    offset_out = pid_batch * stride_out + pid_m * N_SIZE
    
    # Initialize variance accumulator
    var = tl.zeros([1], dtype=tl.float32)
    
    # Compute variance in blocks
    for block_start in range(0, N_SIZE, BLOCK_N_SIZE):
        # Create block mask
        block_mask = block_start + tl.arange(0, BLOCK_N_SIZE) < N_SIZE
        
        # Load x values for this block
        x_block_ptr = x_ptr + offset_x + block_start
        x = tl.load(x_block_ptr, mask=block_mask, other=0.0)
        
        # Accumulate squared values
        var += tl.sum(x * x * block_mask, axis=0)
    
    # Compute RMS statistics
    var = var / N_SIZE
    rstd = 1 / tl.sqrt(var + eps)
    
    # Normalize and apply weight in blocks
    for block_start in range(0, N_SIZE, BLOCK_N_SIZE):
        block_mask = block_start + tl.arange(0, BLOCK_N_SIZE) < N_SIZE
        
        # Load values
        x_block_ptr = x_ptr + offset_x + block_start
        w_block_ptr = rms_w_ptr + block_start
        
        x = tl.load(x_block_ptr, mask=block_mask, other=0.0)
        w = tl.load(w_block_ptr, mask=block_mask, other=1.0)
        
        # Normalize and scale
        out = x * rstd * w
        
        # Store result
        out_block_ptr = output_ptr + offset_out + block_start
        tl.store(out_block_ptr, out, mask=block_mask)

def rmsnorm_triton_wrapper(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6):
    """
    Wrapper function for RMS normalization using Triton kernel
    
    Args:
        x: Input tensor of shape (B, M, N)
        weight: Weight tensor of shape (N,)
        eps: Epsilon for numerical stability
    
    Returns:
        Normalized tensor of same shape as input
    """
    assert x.is_cuda and weight.is_cuda, "Input tensors must be on GPU"
    assert len(x.shape) == 3, "Input tensor must be 3D (B, M, N)"
    assert len(weight.shape) == 1, "Weight tensor must be 1D (N,)"
    
    batch_size, M, N = x.shape
    
    # Ensure tensors are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    
    # Prepare output tensor
    output = torch.empty_like(x)
    
    # Calculate optimal block size
    BLOCK_N_SIZE = min(triton.next_power_of_2(N), 1024)
    
    # Launch kernel
    grid = (batch_size, M)
    rmsnorm_triton[grid](
        x_ptr=x,
        rms_w_ptr=weight,
        output_ptr=output,
        stride_x=M * N,
        stride_w=N,
        stride_out=M * N,
        N_SIZE=N,
        eps=eps,
        BLOCK_N_SIZE=BLOCK_N_SIZE,
        num_warps=4,
    )
    
    return output
