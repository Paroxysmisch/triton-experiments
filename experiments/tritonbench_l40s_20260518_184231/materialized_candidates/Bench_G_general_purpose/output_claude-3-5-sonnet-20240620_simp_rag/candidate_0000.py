import math
import torch
import triton
import triton.language as tl

@triton.jit
def _dequantize_rowwise(
    x_ptr,          # Pointer to input quantized tensor
    state_x,        # Pointer to state tensor containing max values
    output_ptr,     # Pointer to output tensor
    inv_127,        # Scaling factor (1/127)
    n_elements,     # Total number of elements
    BLOCK_SIZE: tl.constexpr,  # Size of each row
    P2: tl.constexpr,          # Next power of 2 after BLOCK_SIZE
):
    # Get the program ID (row index)
    pid = tl.program_id(axis=0)
    
    # Calculate starting offset for this row
    block_start = pid * BLOCK_SIZE
    
    # Create offset array for the entire block
    arange = tl.arange(0, P2)
    offsets = block_start + arange
    
    # Create mask for valid elements in the row
    row_mask = arange < BLOCK_SIZE
    
    # Load input values
    x = tl.load(x_ptr + offsets, mask=row_mask)
    
    # Load max value for this row
    max_val = tl.load(state_x + pid)
    
    # Perform dequantization: output = max_val * x * (1/127)
    output = max_val * x * inv_127
    
    # Store the result
    tl.store(output_ptr + offsets, output, mask=row_mask)

def dequantize_rowwise(x: torch.Tensor, state_x: torch.Tensor):
    """
    Wrapper function for row-wise dequantization
    Args:
        x: Input quantized tensor
        state_x: Tensor containing max values for each row
    Returns:
        Dequantized tensor
    """
    # Initialize output tensor
    output = torch.empty(*x.shape, device=x.device, dtype=torch.float16)
    
    # Calculate next power of 2 for the number of columns
    P2 = int(2 ** (math.ceil(math.log2(x.shape[1]))))
    
    # Verify tensors are on GPU
    assert x.is_cuda and output.is_cuda
    
    # Get total number of elements
    n_elements = output.numel()
    
    # Define grid for kernel launch (one block per row)
    grid = lambda meta: (x.shape[0],)
    
    # Launch kernel
    _dequantize_rowwise[grid](
        x, state_x, output, 
        1./127, n_elements, 
        BLOCK_SIZE=x.shape[1], 
        P2=P2
    )
    
    return output
