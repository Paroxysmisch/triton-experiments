import triton
import triton.language as tl

@triton.jit
def row_mean_kernel(input_ptr, output_ptr, row_size, stride, numel, BLOCK_SIZE: tl.constexpr):
    # Position of the current block in the grid
    row_idx = tl.program_id(0)
    
    # Create a range for the current block
    offsets = row_idx * stride + tl.arange(0, BLOCK_SIZE)
    
    # Load input data
    data = tl.load(input_ptr + offsets, mask=offsets < numel, other=0.0)
    
    # Compute the sum
    row_sum = tl.sum(data, axis=0)
    
    # Compute the mean
    row_mean = row_sum / row_size
    
    # Store the result
    tl.store(output_ptr + row_idx, row_mean)


import torch

def triton_row_mean(input, dim, keepdim=False, dtype=None, out=None):
    # Cast input to the desired dtype if specified
    if dtype is not None:
        input = input.to(dtype)
    
    # Handle dimensions: convert dim to a tuple if it's not already
    if isinstance(dim, int):
        dim = (dim,)
    
    # Compute the shape of the output tensor
    output_shape = list(input.shape)
    for d in dim:
        if keepdim:
            output_shape[d] = 1
        else:
            output_shape.pop(d)
    
    # Prepare output tensor
    if out is None:
        out = torch.empty(output_shape, dtype=input.dtype, device=input.device)
    
    # Flatten dimensions if necessary
    input_flat = input
    for d in sorted(dim, reverse=True):
        input_flat = input_flat.flatten(start_dim=d)
    
    # Get row size (product of dimensions to be reduced)
    row_size = torch.prod(torch.tensor([input.shape[d] for d in dim], dtype=torch.int32))
    
    # Define block size for Triton kernel
    BLOCK_SIZE = 1024  # You may adjust this based on hardware and input size
    
    # Launch Triton kernel
    num_rows = input_flat.shape[0]
    row_mean_kernel[(num_rows,)](input_flat, out, row_size, input_flat.stride(0), input_flat.numel(), BLOCK_SIZE=BLOCK_SIZE)
    
    # Reshape output to match expected shape
    if keepdim:
        out = out.view(output_shape)
    else:
        out = out.squeeze(dim)
    
    return out
