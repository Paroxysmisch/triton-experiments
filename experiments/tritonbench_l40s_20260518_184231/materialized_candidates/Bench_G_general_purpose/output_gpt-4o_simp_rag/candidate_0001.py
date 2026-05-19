import math
import torch
import triton
import triton.language as tl

@triton.jit
def _dequantize_rowwise(
    x_ptr,
    state_x,
    output_ptr,
    inv_127,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    P2: tl.constexpr,
):
    # Get the program ID for the current block of rows
    pid = tl.program_id(axis=0)
    
    # Calculate the start index of the block
    block_start = pid * BLOCK_SIZE
    
    # Create a range of indices for the current block
    arange = tl.arange(0, P2)
    
    # Calculate the offsets for loading/storing data
    offsets = block_start + arange
    
    # Create a mask to handle elements within the valid range
    row_mask = arange < BLOCK_SIZE
    
    # Load the input data for the current row with masking
    x = tl.load(x_ptr + offsets, mask=row_mask, other=0.0)
    
    # Load the maximum value for the current row
    max_val = tl.load(state_x + pid)
    
    # Perform the dequantization
    output = max_val * x * inv_127
    
    # Store the result in the output tensor with masking
    tl.store(output_ptr + offsets, output, mask=row_mask)

def dequantize_rowwise(x: torch.Tensor, state_x: torch.Tensor):
    # Prepare the output tensor
    output = torch.empty_like(x, dtype=torch.float32)
    
    # Calculate the nearest power of 2 for the number of columns
    P2 = int(2 ** (math.ceil(math.log2(x.shape[1]))))
    
    # Ensure input and output tensors are on the GPU
    assert x.is_cuda and output.is_cuda
    
    # Calculate the number of elements to process
    n_elements = output.numel()
    
    # Define the grid for Triton kernel execution
    grid = lambda meta: (x.shape[0],)
    
    # Launch the Triton kernel
    _dequantize_rowwise[grid](
        x, state_x, output, 1./127, n_elements, 
        BLOCK_SIZE=x.shape[1], P2=P2
    )
    
    return output
