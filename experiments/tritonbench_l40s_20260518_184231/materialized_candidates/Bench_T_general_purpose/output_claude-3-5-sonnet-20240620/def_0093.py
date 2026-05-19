import torch
import triton
import triton.language as tl

@triton.jit
def softmax_log_kernel(
    output_ptr,
    input_ptr,
    stride_dim,
    stride_batch,
    n_dim,
    n_batch,
    BLOCK_SIZE: tl.constexpr,
):
    # Position of element
    pid = tl.program_id(0)
    
    # Batch index
    batch_idx = pid // (n_dim // BLOCK_SIZE)
    
    # Starting offset for this program instance
    dim_offset = (pid % (n_dim // BLOCK_SIZE)) * BLOCK_SIZE
    
    # Compute base pointers
    batch_offset = batch_idx * stride_batch
    input_block_ptr = input_ptr + batch_offset + dim_offset
    output_block_ptr = output_ptr + batch_offset + dim_offset
    
    # Load input block
    mask = tl.arange(0, BLOCK_SIZE) < (n_dim - dim_offset)
    x = tl.load(input_block_ptr + tl.arange(0, BLOCK_SIZE) * stride_dim, mask=mask)
    
    # Apply log
    x = tl.log(x)
    
    # Compute max for numerical stability
    x_max = tl.max(x, axis=0)
    
    # Compute exponentials
    x_exp = tl.exp(x - x_max)
    
    # Compute sum
    x_sum = tl.sum(x_exp, axis=0)
    
    # Normalize
    output = x_exp / x_sum
    
    # Store result
    tl.store(output_block_ptr + tl.arange(0, BLOCK_SIZE) * stride_dim, output, mask=mask)

def softmax_log(input: torch.Tensor, dim: int = -1, dtype: torch.dtype = None) -> torch.Tensor:
    """
    Applies natural logarithm element-wise followed by softmax along specified dimension.
    
    Args:
        input (Tensor): Input tensor
        dim (int): Dimension along which softmax will be computed (default: -1)
        dtype (torch.dtype, optional): Desired output dtype (default: None)
    
    Returns:
        Tensor: Output tensor of same shape as input
    """
    # Handle negative dim
    if dim < 0:
        dim = input.dim() + dim
    
    # Cast input if dtype specified
    if dtype is not None:
        input = input.to(dtype)
    
    # Ensure input is contiguous and on GPU
    input = input.contiguous()
    if not input.is_cuda:
        input = input.cuda()
    
    # Get input shape and size
    shape = input.shape
    n_dim = shape[dim]
    
    # Calculate batch size (product of all other dimensions)
    n_batch = input.numel() // n_dim
    
    # Compute strides
    stride_dim = input.stride()[dim]
    stride_batch = input.stride()[0] if dim != 0 else input.stride()[1]
    
    # Determine block size (power of 2 <= n_dim)
    BLOCK_SIZE = triton.next_power_of_2(min(n_dim, 512))
    
    # Create output tensor
    output = torch.empty_like(input)
    
    # Launch kernel
    grid = (triton.cdiv(n_dim, BLOCK_SIZE) * n_batch,)
    softmax_log_kernel[grid](
        output,
        input,
        stride_dim,
        stride_batch,
        n_dim,
        n_batch,
        BLOCK_SIZE,
    )
    
    return output
