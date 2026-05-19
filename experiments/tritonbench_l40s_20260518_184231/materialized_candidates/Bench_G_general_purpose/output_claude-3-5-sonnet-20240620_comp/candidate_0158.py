import torch
import triton
import triton.language as tl

@triton.jit
def rmsnorm_triton(
    x_ptr, rms_w_ptr, output_ptr,
    stride_x_b, stride_x_m, stride_x_n,
    stride_w,
    stride_out_b, stride_out_m, stride_out_n,
    N_SIZE: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_N_SIZE: tl.constexpr,
):
    # Get program ID
    pid_batch = tl.program_id(0)
    pid_m = tl.program_id(1)
    
    # Compute base pointers
    x_base_ptr = x_ptr + pid_batch * stride_x_b + pid_m * stride_x_m
    w_base_ptr = rms_w_ptr
    output_base_ptr = output_ptr + pid_batch * stride_out_b + pid_m * stride_out_m
    
    # Initialize variance accumulator
    var = tl.zeros([1], dtype=tl.float32)
    
    # First pass: compute variance
    for block_start_n in range(0, N_SIZE, BLOCK_N_SIZE):
        # Create block mask
        block_mask = block_start_n + tl.arange(0, BLOCK_N_SIZE) < N_SIZE
        
        # Load x block
        x_block_ptr = x_base_ptr + block_start_n * stride_x_n
        x = tl.load(x_block_ptr, mask=block_mask, other=0.0)
        
        # Accumulate squared values
        var += tl.sum(x * x * block_mask, axis=0)
    
    # Compute RMS statistics
    var = var / N_SIZE
    rstd = 1.0 / tl.sqrt(var + eps)
    
    # Second pass: normalize and scale
    for block_start_n in range(0, N_SIZE, BLOCK_N_SIZE):
        # Create block mask
        block_mask = block_start_n + tl.arange(0, BLOCK_N_SIZE) < N_SIZE
        
        # Load x and weight blocks
        x_block_ptr = x_base_ptr + block_start_n * stride_x_n
        w_block_ptr = w_base_ptr + block_start_n * stride_w
        
        x = tl.load(x_block_ptr, mask=block_mask, other=0.0)
        w = tl.load(w_block_ptr, mask=block_mask, other=0.0)
        
        # Normalize and scale
        output = x * rstd * w
        
        # Store result
        output_block_ptr = output_base_ptr + block_start_n * stride_out_n
        tl.store(output_block_ptr, output, mask=block_mask)

def rmsnorm_triton_wrapper(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6):
    """
    Wrapper function for RMSNorm Triton kernel
    Args:
        x: Input tensor of shape (B, M, N)
        weight: Weight tensor of shape (N,)
        eps: Epsilon for numerical stability
    Returns:
        Output tensor of shape (B, M, N)
    """
    assert x.dim() == 3, "Input tensor must be 3D (B, M, N)"
    assert weight.dim() == 1, "Weight tensor must be 1D (N,)"
    assert x.size(-1) == weight.size(0), "Last dimension of input must match weight size"
    
    batch_size, M, N = x.shape
    
    # Ensure tensors are contiguous and in correct format
    x = x.contiguous()
    weight = weight.contiguous()
    
    # Create output tensor
    output = torch.empty_like(x)
    
    # Configure block size for efficient memory access
    BLOCK_N_SIZE = min(triton.next_power_of_2(N), 128)
    
    # Launch kernel
    grid = (batch_size, M)
    rmsnorm_triton[grid](
        x_ptr=x.data_ptr(),
        rms_w_ptr=weight.data_ptr(),
        output_ptr=output.data_ptr(),
        stride_x_b=x.stride(0),
        stride_x_m=x.stride(1),
        stride_x_n=x.stride(2),
        stride_w=weight.stride(0),
        stride_out_b=output.stride(0),
        stride_out_m=output.stride(1),
        stride_out_n=output.stride(2),
        N_SIZE=N,
        eps=eps,
        BLOCK_N_SIZE=BLOCK_N_SIZE,
    )
    
    return output
