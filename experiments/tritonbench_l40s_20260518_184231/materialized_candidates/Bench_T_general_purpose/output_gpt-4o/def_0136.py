import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(input_ptr, output_ptr, n_elements, dim_stride, BLOCK_SIZE: tl.constexpr):
    # Calculate the offset for the current block
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Load data from the input tensor
    input_data = tl.load(input_ptr + offsets * dim_stride, mask=offsets < n_elements, other=-float('inf'))
    
    # Compute the maximum value for numerical stability
    max_val = tl.max(input_data, axis=0)
    
    # Subtract max value and exponentiate
    input_data = tl.exp(input_data - max_val)
    
    # Compute the sum of exponentials
    sum_exp = tl.sum(input_data, axis=0)
    
    # Normalize the values
    softmax_output = input_data / sum_exp
    
    # Store the result in the output tensor
    tl.store(output_ptr + offsets * dim_stride, softmax_output, mask=offsets < n_elements)

def softmax(input, dim, dtype=None):
    # Ensure input is a torch tensor
    if not isinstance(input, torch.Tensor):
        raise TypeError("Input must be a torch.Tensor")
    
    # Move input to the specified dtype if provided
    if dtype is not None:
        input = input.to(dtype)
    
    # Get the shape and number of elements along the specified dimension
    n_elements = input.shape[dim]
    dim_stride = input.stride(dim)
    
    # Allocate output tensor
    output = torch.empty_like(input)
    
    # Launch the Triton kernel
    grid = (triton.cdiv(n_elements, 1024),)
    softmax_kernel[grid](input.data_ptr(), output.data_ptr(), n_elements, dim_stride, BLOCK_SIZE=1024)
    
    return output
