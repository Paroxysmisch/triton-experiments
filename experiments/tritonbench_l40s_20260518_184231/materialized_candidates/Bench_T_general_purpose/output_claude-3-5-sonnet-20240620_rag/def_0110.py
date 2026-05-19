import triton
import triton.language as tl
import torch

@triton.jit
def exp_mean_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    stride,
    BLOCK_SIZE: tl.constexpr,
):
    """Kernel that computes exponential and mean of a tensor.
    
    Args:
        input_ptr: Pointer to input tensor
        output_ptr: Pointer to output tensor
        n_elements: Total number of elements
        stride: Stride for accessing elements
        BLOCK_SIZE: Size of the block for parallel processing
    """
    # Get program ID
    pid = tl.program_id(0)
    
    # Compute block start/end
    block_start = pid * BLOCK_SIZE
    block_end = tl.minimum(block_start + BLOCK_SIZE, n_elements)
    
    # Create offset array for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input values
    x = tl.load(input_ptr + offsets * stride, mask=mask)
    
    # Compute exponential
    exp_vals = tl.exp(x)
    
    # Compute sum for this block
    block_sum = tl.sum(tl.where(mask, exp_vals, 0.0))
    
    # Store result
    tl.atomic_add(output_ptr, block_sum)

def exp_mean(input, dim=None, keepdim=False, dtype=None, out=None) -> torch.Tensor:
    """Applies exponential function to input tensor and computes mean.
    
    Args:
        input: Input tensor
        dim: Dimension along which to compute mean (None for global mean)
        keepdim: Whether to keep reduced dimensions
        dtype: Output dtype
        out: Output tensor
        
    Returns:
        Tensor containing mean of exponential values
    """
    if dtype is None:
        dtype = input.dtype
    
    # Handle global mean case (dim=None)
    if dim is None:
        # Initialize output tensor
        if out is None:
            out = torch.zeros(1, device=input.device, dtype=dtype)
        else:
            out.zero_()
            
        # Launch kernel
        n_elements = input.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        exp_mean_kernel[grid](
            input_ptr=input.data_ptr(),
            output_ptr=out.data_ptr(),
            n_elements=n_elements,
            stride=1,
            BLOCK_SIZE=1024,
        )
        
        # Compute mean
        out.div_(n_elements)
        
        return out.squeeze() if not keepdim else out
        
    else:
        # For specific dimensions, we need to handle the reduction differently
        # First apply exponential
        exp_input = torch.exp(input)
        # Then compute mean along specified dimensions
        return torch.mean(exp_input, dim=dim, keepdim=keepdim, dtype=dtype, out=out)
