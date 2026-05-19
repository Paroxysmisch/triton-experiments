import triton
import triton.language as tl

@triton.jit
def softmax_mul_kernel(
    input_ptr,  # Pointer to the input tensor
    other_ptr,  # Pointer to the other tensor or number
    output_ptr, # Pointer to the output tensor
    stride_input,  # Stride of the input tensor
    stride_other,  # Stride of the other tensor (or 0 if other is a scalar)
    stride_output, # Stride of the output tensor
    n_elements,    # Number of elements in the input tensor
    dim_size,      # Size of the dimension along which softmax is applied
    BLOCK_SIZE: tl.constexpr
):
    # Compute the block ID
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    # Compute the block of elements to process
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load the input and other tensors
    input_vec = tl.load(input_ptr + offsets, mask=mask)
    other_vec = tl.load(other_ptr + offsets, mask=mask) if stride_other != 0 else other_ptr

    # Compute the max value for numerical stability
    max_val = tl.max(input_vec, axis=0)
    input_vec = input_vec - max_val

    # Compute the exponential values
    exp_val = tl.exp(input_vec)

    # Compute the sum of exponentials
    sum_exp = tl.sum(exp_val, axis=0)

    # Compute the softmax values
    softmax_val = exp_val / sum_exp

    # Compute the final output
    output_vec = softmax_val * other_vec

    # Store the result
    tl.store(output_ptr + offsets, output_vec, mask=mask)

import torch
import triton
import triton.language as tl

def softmax_mul(input, other, dim, dtype=None, out=None):
    # Ensure input is a tensor
    if not isinstance(input, torch.Tensor):
        raise TypeError("input must be a torch.Tensor")
    
    # Ensure other is a tensor or a number
    if not (isinstance(other, torch.Tensor) or isinstance(other, (int, float))):
        raise TypeError("other must be a torch.Tensor or a number")
    
    # Ensure dim is an integer
    if not isinstance(dim, int):
        raise TypeError("dim must be an integer")
    
    # Ensure dim is within the input tensor's dimensions
    if dim < 0 or dim >= input.dim():
        raise ValueError(f"dim must be within the range of input tensor dimensions (0 to {input.dim() - 1})")
    
    # Cast input to the specified dtype if provided
    if dtype is not None:
        input = input.to(dtype)
    
    # Determine the size of the dimension along which softmax is applied
    dim_size = input.size(dim)
    
    # Flatten the input tensor along the specified dimension
    input_flattened = input.flatten(start_dim=dim, end_dim=-1)
    n_elements = input_flattened.numel()
    
    # Flatten the other tensor if it is a tensor
    if isinstance(other, torch.Tensor):
        other_flattened = other.flatten(start_dim=dim, end_dim=-1)
    else:
        other_flattened = other
    
    # Allocate the output tensor if not provided
    if out is None:
        out = torch.empty_like(input_flattened, device=input.device, dtype=input.dtype)
    else:
        if out.shape != input_flattened.shape:
            raise ValueError("out tensor must have the same shape as the input tensor")
    
    # Define the grid and block sizes
    BLOCK_SIZE = 1024
    grid = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch the Triton kernel
    softmax_mul_kernel[grid, BLOCK_SIZE](
        input_flattened,  # Pointer to the input tensor
        other_flattened,  # Pointer to the other tensor or number
        out,              # Pointer to the output tensor
        input_flattened.stride(0),  # Stride of the input tensor
        other_flattened.stride(0) if isinstance(other, torch.Tensor) else 0,  # Stride of the other tensor (or 0 if other is a scalar)
        out.stride(0),    # Stride of the output tensor
        n_elements,       # Number of elements in the input tensor
        dim_size,         # Size of the dimension along which softmax is applied
        BLOCK_SIZE        # Block size
    )
    
    # Reshape the output tensor to match the input tensor's shape
    out = out.view_as(input)
    
    return out
