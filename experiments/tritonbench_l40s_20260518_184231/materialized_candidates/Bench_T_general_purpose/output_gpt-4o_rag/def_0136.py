import torch
import triton
import triton.language as tl
from triton.runtime import driver

@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_stride, output_stride, n_elements, dim_size, BLOCK_SIZE: tl.constexpr):
    # Starting position for the program
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    block_end = min(block_start + BLOCK_SIZE, n_elements)
    
    # Loading the input slice (dimension `dim`)
    input_ptrs = input_ptr + block_start + tl.arange(0, BLOCK_SIZE)
    mask = block_start + tl.arange(0, BLOCK_SIZE) < n_elements
    data = tl.load(input_ptrs, mask=mask, other=0.0)  # Load slice with padding as zero
    
    # Subtract maximum for numerical stability
    max_val = tl.max(data)
    stabilized_data = data - max_val
    
    # Exponentiation for softmax
    exp_data = tl.exp(stabilized_data)
    
    # Sum along the dimension for softmax denominator
    sum_exp = tl.sum(exp_data)
    
    # Compute softmax (numerator / denominator)
    softmax_output = exp_data / sum_exp
    
    # Store the result
    output_ptrs = output_ptr + block_start + tl.arange(0, BLOCK_SIZE)
    tl.store(output_ptrs, softmax_output, mask=mask)

def softmax(input, dim, dtype=None) -> torch.Tensor:
    # Ensure the input tensor is contiguous for stride computation
    input = input.contiguous()

    # Determine the shape of the input tensor
    n_elements = input.shape[dim]
    shape = input.shape
    dim_size = shape[dim]
    
    # We are applying softmax across the specified dimension
    if dim == 0:
        input_stride = input.stride(0)
        output_stride = input.stride(0)
    else:
        input_stride = input.stride(dim)
        output_stride = input.stride(dim)
    
    # Create a tensor to store the output
    output = torch.empty_like(input, dtype=dtype if dtype is not None else input.dtype)

    # Block size for processing; you can optimize this depending on the tensor's shape
    BLOCK_SIZE = 64  # For example, can be adjusted based on hardware

    # Launch the kernel
    grid = (input.shape[0] + BLOCK_SIZE - 1) // BLOCK_SIZE  # Adjust grid size based on input shape
    softmax_kernel[grid](
        output,
        input,
        input_stride,
        output_stride,
        n_elements,
        dim_size,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output
