import torch
import triton
import triton.language as tl

@triton.jit
def _dequantize_rowwise(
    x_ptr,          # Pointer to input tensor (quantized)
    state_x_ptr,    # Pointer to state tensor (max values)
    output_ptr,     # Pointer to output tensor
    inv_127,        # Precomputed 1/127
    n_elements,     # Total number of elements
    BLOCK_SIZE: tl.constexpr,  # Size of block for processing
    P2: tl.constexpr,  # Power of 2 nearest to number of columns
):
    # Get the program ID for the current row
    pid = tl.program_id(0)
    
    # Calculate starting position for this block
    block_start = pid * P2
    
    # Create offset range for the current block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for valid memory accesses
    mask = offsets < n_elements
    
    # Load quantized values
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Load max value for the current row
    max_val = tl.load(state_x_ptr + pid)
    
    # Dequantize values
    output = x * max_val * inv_127
    
    # Store results
    tl.store(output_ptr + offsets, output, mask=mask)

def dequantize_rowwise(x, state_x):
    """
    Wrapper function for row-wise dequantization using Triton
    
    Args:
        x: Input tensor (quantized)
        state_x: State tensor containing max values
    
    Returns:
        Dequantized tensor
    """
    assert x.is_cuda and state_x.is_cuda, "Input tensors must be on GPU"
    
    # Get tensor dimensions
    n_rows, n_cols = x.shape
    
    # Calculate nearest power of 2 for efficient memory access
    P2 = triton.next_power_of_2(n_cols)
    
    # Prepare output tensor
    output = torch.empty_like(x, dtype=torch.float32, device=x.device)
    
    # Precompute inverse of 127
    inv_127 = 1.0 / 127.0
    
    # Configure grid for kernel launch
    grid = (n_rows,)
    
    # Launch kernel
    _dequantize_rowwise[grid](
        x.data_ptr(),
        state_x.data_ptr(),
        output.data_ptr(),
        inv_127,
        n_cols,
        BLOCK_SIZE=P2,
        P2=P2,
    )
    
    return output
