import triton
import triton.language as tl
import torch
import math

@triton.jit
def gelu_min_kernel(
    x_ptr,  # pointer to input tensor
    out_ptr,  # pointer to output tensor
    indices_ptr,  # pointer to indices tensor
    stride_xm, stride_xn,  # strides for accessing input
    stride_om, stride_on,  # strides for accessing output
    size_m, size_n,  # input tensor dimensions
    BLOCK_SIZE: tl.constexpr,  # block size for parallelization
    approximate: tl.constexpr,  # GELU approximation method
    reduce_dim: tl.constexpr,  # dimension to reduce along
):
    # Position of elements processed by this program
    pid = tl.program_id(0)
    
    # Initialize variables for reduction
    if reduce_dim == 0:
        row = 0
        col = pid
    else:
        row = pid
        col = 0
    
    # Load block of elements
    mask = tl.arange(0, BLOCK_SIZE) < (size_n if reduce_dim == 0 else size_m)
    
    # Initialize minimum value and index
    min_val = float('inf')
    min_idx = 0
    
    # Compute GELU and reduction
    for i in range(0, size_n if reduce_dim == 0 else size_m, BLOCK_SIZE):
        # Compute offsets and load data
        offs = tl.arange(0, BLOCK_SIZE)
        curr_mask = mask & (i + offs < (size_n if reduce_dim == 0 else size_m))
        
        if reduce_dim == 0:
            x = tl.load(x_ptr + (row + offs) * stride_xm + col * stride_xn, mask=curr_mask)
        else:
            x = tl.load(x_ptr + row * stride_xm + (col + offs) * stride_xn, mask=curr_mask)
        
        # Compute GELU
        if approximate == 0:  # exact
            # GELU(x) = x * Φ(x)
            sqrt2 = 1.4142135623730951
            cdf = 0.5 * (1.0 + tl.erf(x / sqrt2))
            gelu_val = x * cdf
        else:  # tanh approximation
            # GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
            sqrt2_pi = 0.7978845608028654
            tanh_arg = sqrt2_pi * (x + 0.044715 * x * x * x)
            gelu_val = 0.5 * x * (1.0 + tl.tanh(tanh_arg))
        
        # Update minimum
        curr_min = tl.min(gelu_val, axis=0)
        curr_idx = tl.argmin(gelu_val, axis=0)
        
        if curr_min < min_val:
            min_val = curr_min
            min_idx = i + curr_idx

    # Store results
    if reduce_dim == 0:
        tl.store(out_ptr + col, min_val)
        tl.store(indices_ptr + col, min_idx)
    else:
        tl.store(out_ptr + row, min_val)
        tl.store(indices_ptr + row, min_idx)

def gelu_min(input, approximate='none', dim=None, keepdim=False, out=None):
    """
    Applies GELU activation followed by minimum reduction along specified dimension.
    
    Args:
        input (Tensor): Input tensor
        approximate (str): GELU approximation method ('none' or 'tanh')
        dim (int, optional): Dimension to reduce along
        keepdim (bool): Whether to keep reduced dimension
        out (Tuple[Tensor, Tensor], optional): Output tensor and indices
    
    Returns:
        Tuple[Tensor, LongTensor] or Tensor: Minimum values and indices if dim specified,
                                           otherwise minimum value tensor
    """
    # Input validation
    if approximate not in ['none', 'tanh']:
        raise ValueError("approximate must be 'none' or 'tanh'")
    
    # Handle scalar input
    if input.dim() == 0:
        return input.clone()
    
    # Prepare output tensors
    if out is None:
        if dim is None:
            output = torch.empty((), device=input.device, dtype=input.dtype)
            indices = torch.empty((), device=input.device, dtype=torch.long)
        else:
            output_shape = list(input.shape)
            if not keepdim:
                output_shape.pop(dim)
            output = torch.empty(output_shape, device=input.device, dtype=input.dtype)
            indices = torch.empty(output_shape, device=input.device, dtype=torch.long)
    else:
        output, indices = out
    
    # Configure kernel parameters
    BLOCK_SIZE = 1024
    approximate_val = 1 if approximate == 'tanh' else 0
    
    # Launch kernel
    def grid(meta):
        return (input.size(1) if dim == 0 else input.size(0),)
    
    gelu_min_kernel[grid](
        input, output, indices,
        input.stride(0), input.stride(1),
        output.stride(0) if output.dim() > 1 else 0,
        output.stride(1) if output.dim() > 1 else 0,
        input.size(0), input.size(1),
        BLOCK_SIZE,
        approximate_val,
        0 if dim is None else dim,
    )
    
    if dim is None:
        return output
    else:
        return torch.return_types.min(values=output, indices=indices)
