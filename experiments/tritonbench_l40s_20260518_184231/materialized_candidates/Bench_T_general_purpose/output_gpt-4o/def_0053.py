import triton
import triton.language as tl

@triton.jit
def mul_relu_kernel(
    input_ptr, other_ptr, output_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr
):
    # Define the block index
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Load input and other
    input = tl.load(input_ptr + offsets, mask=offsets < n_elements, other=0.0)
    other = tl.load(other_ptr + offsets, mask=offsets < n_elements, other=0.0)
    
    # Perform element-wise multiplication
    result = input * other
    
    # Apply ReLU
    result = tl.where(result > 0, result, 0.0)
    
    # Store the result
    tl.store(output_ptr + offsets, result, mask=offsets < n_elements)


import torch
import triton

def mul_relu(input, other, inplace=False, out=None):
    # Ensure `other` is a tensor
    if isinstance(other, (int, float)):
        other = torch.tensor(other, dtype=input.dtype, device=input.device)
    
    # Determine the output tensor
    if out is None:
        out = torch.empty_like(input)
    
    # Check if in-place operation is required
    if inplace:
        out = input
    
    # Get the number of elements
    n_elements = input.numel()
    
    # Launch the Triton kernel
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    BLOCK_SIZE = 1024  # You can adjust this block size as needed
    
    mul_relu_kernel[grid](
        input_ptr=input,
        other_ptr=other,
        output_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out
