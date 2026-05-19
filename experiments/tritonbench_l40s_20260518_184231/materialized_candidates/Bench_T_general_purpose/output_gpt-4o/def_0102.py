import triton
import triton.language as tl

@triton.jit
def softmax_mul_kernel(
    input_ptr, other_ptr, output_ptr, n_elements,
    dim_stride, dim_size, other_is_scalar, other_value,
    BLOCK_SIZE: tl.constexpr
):
    # Get the program id
    pid = tl.program_id(0)
    
    # Create a range for the block
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask for valid indices
    mask = offsets < n_elements
    
    # Load input data
    input = tl.load(input_ptr + offsets * dim_stride, mask=mask, other=0.0)
    
    # Compute the max for numerical stability
    max_input = tl.max(input, axis=0)
    
    # Subtract max and exponentiate
    exp_input = tl.exp(input - max_input)
    
    # Compute the sum of exponentials
    sum_exp = tl.sum(exp_input, axis=0)
    
    # Compute softmax
    softmax_output = exp_input / sum_exp
    
    # Load or broadcast other
    if other_is_scalar:
        other = other_value
    else:
        other = tl.load(other_ptr + offsets * dim_stride, mask=mask, other=1.0)
    
    # Compute the output
    output = softmax_output * other
    
    # Store the result
    tl.store(output_ptr + offsets * dim_stride, output, mask=mask)

import torch

def softmax_mul(input, other, dim, dtype=None, out=None):
    # Check if other is a scalar
    other_is_scalar = isinstance(other, (int, float))
    
    # Convert input to the specified dtype if provided
    if dtype is not None:
        input = input.to(dtype)
        if not other_is_scalar:
            other = other.to(dtype)
    
    # Create output tensor if not provided
    if out is None:
        out = torch.empty_like(input)
    
    # Get the size and stride of the specified dimension
    dim_size = input.size(dim)
    dim_stride = input.stride(dim)
    
    # Flatten the input for Triton processing
    n_elements = input.numel()
    
    # Launch the Triton kernel
    BLOCK_SIZE = 1024  # You can choose an appropriate block size
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    softmax_mul_kernel[grid](
        input_ptr=input.data_ptr(),
        other_ptr=other if not other_is_scalar else 0,
        output_ptr=out.data_ptr(),
        n_elements=n_elements,
        dim_stride=dim_stride,
        dim_size=dim_size,
        other_is_scalar=other_is_scalar,
        other_value=other if other_is_scalar else 0,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out
