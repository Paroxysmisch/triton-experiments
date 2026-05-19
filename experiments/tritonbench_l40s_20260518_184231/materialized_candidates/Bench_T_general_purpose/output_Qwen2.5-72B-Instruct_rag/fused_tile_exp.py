import torch
import triton
import triton.language as tl

@triton.jit
def fused_tile_exp_kernel(
    output_ptr,  # Pointer to the output tensor
    input_ptr,   # Pointer to the input tensor
    input_shape, # Shape of the input tensor
    input_strides, # Strides of the input tensor
    output_strides, # Strides of the output tensor
    dims,        # Tiling dimensions
    n_elements,  # Total number of elements in the output tensor
    BLOCK_SIZE: tl.constexpr
):
    # Get the program ID
    pid = tl.program_id(0)
    # Calculate the range of elements this program will handle
    block_start = pid * BLOCK_SIZE
    block_end = min(block_start + BLOCK_SIZE, n_elements)
    
    # Iterate over the elements in the block
    for i in range(block_start, block_end):
        # Calculate the original index in the input tensor
        original_index = 0
        temp_i = i
        for dim in range(len(input_shape)):
            stride = output_strides[dim]
            size = input_shape[dim]
            tile = dims[dim]
            original_index += (temp_i // (stride * tile)) % size * input_strides[dim]
            temp_i %= stride * tile
        
        # Load the element from the input tensor
        input_val = tl.load(input_ptr + original_index)
        
        # Apply the exponential function
        output_val = tl.exp(input_val)
        
        # Store the result in the output tensor
        tl.store(output_ptr + i, output_val)

def fused_tile_exp(input, dims, *, out=None):
    # Ensure dims has the same number of dimensions as input
    input_shape = input.shape
    if len(dims) < len(input_shape):
        dims = (1,) * (len(input_shape) - len(dims)) + dims
    
    # Calculate the shape of the output tensor
    output_shape = tuple(s * d for s, d in zip(input_shape, dims))
    
    # Allocate the output tensor if not provided
    if out is None:
        out = torch.empty(output_shape, dtype=input.dtype, device=input.device)
    else:
        assert out.shape == output_shape, "Output tensor shape does not match the expected shape"
    
    # Determine the strides for the input and output tensors
    input_strides = input.stride()
    output_strides = out.stride()
    
    # Determine the number of elements in the output tensor
    n_elements = out.numel()
    
    # Determine the block size
    BLOCK_SIZE = 1024  # Adjust this value based on your GPU architecture
    
    # Determine the number of programs (blocks) needed
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch the kernel
    fused_tile_exp_kernel[(num_programs,)](
        out.data_ptr(),
        input.data_ptr(),
        input_shape,
        input_strides,
        output_strides,
        dims,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

import torch

# Create a simple input tensor
input = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32, device='cuda')

# Define the tiling dimensions
dims = (2, 3)

# Call the fused_tile_exp function
output = fused_tile_exp(input, dims)

# Print the output tensor
print(output)
