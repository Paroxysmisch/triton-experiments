import triton
import triton.language as tl
import torch

@triton.jit
def logsumexp_fwd_kernel(
    output_ptr, input_ptr, scale_ptr,
    stride_xb, stride_xd,  # strides for navigating input tensor
    stride_oz,  # stride for output tensor
    n_cols,  # size of reduction dimension (D)
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(0)
    
    # Compute the starting offset for this program instance
    offset_x = pid * stride_xb
    
    # Initialize the maximum value and accumulator
    max_val = -float('inf')
    acc = 0.0
    
    # Load scale factor if provided
    scale = 1.0
    if scale_ptr is not None:
        scale = tl.load(scale_ptr)
    
    # First pass: find maximum
    for i in range(0, n_cols, BLOCK_SIZE):
        # Create block mask for bounds checking
        mask = i + tl.arange(0, BLOCK_SIZE) < n_cols
        
        # Load input values
        x = tl.load(input_ptr + offset_x + i * stride_xd, mask=mask, other=-float('inf'))
        x = x * scale
        
        # Update maximum
        max_val = tl.maximum(max_val, tl.max(x, axis=0))
    
    # Second pass: compute exp and sum
    for i in range(0, n_cols, BLOCK_SIZE):
        mask = i + tl.arange(0, BLOCK_SIZE) < n_cols
        
        # Load and scale input values
        x = tl.load(input_ptr + offset_x + i * stride_xd, mask=mask, other=-float('inf'))
        x = x * scale
        
        # Compute exp(x - max_val) and accumulate
        acc += tl.sum(tl.exp(x - max_val) * mask, axis=0)
    
    # Compute final result: max_val + log(sum(exp(x - max_val)))
    output = max_val + tl.log(acc)
    
    # Store the result
    tl.store(output_ptr + pid * stride_oz, output)

def logsumexp_fwd(x: torch.Tensor, scale: torch.Tensor = None, dtype: torch.dtype = None) -> torch.Tensor:
    """
    Compute log(sum(exp(x))) over the last dimension.
    
    Args:
        x: Input tensor
        scale: Optional scale factor
        dtype: Optional output dtype
    
    Returns:
        Result of log-sum-exp operation
    """
    # Handle input tensor shape
    *batch_dims, D = x.shape
    batch_size = 1
    for dim in batch_dims:
        batch_size *= dim
    
    # Reshape input to 2D
    x_2d = x.reshape(batch_size, D)
    
    # Determine block size (power of 2 that's close to D)
    BLOCK_SIZE = triton.next_power_of_2(min(D, 1024))
    
    # Create output tensor
    z = torch.empty(batch_size, device=x.device, dtype=x.dtype)
    
    # Launch kernel
    grid = (batch_size,)
    logsumexp_fwd_kernel[grid](
        z, x_2d, scale,
        x_2d.stride(0), x_2d.stride(1),
        z.stride(0),
        D,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Reshape output back to original batch dimensions
    z = z.reshape(*batch_dims)
    
    # Cast to requested dtype if specified
    if dtype is not None:
        z = z.to(dtype)
    
    return z
