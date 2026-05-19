import triton
import triton.language as tl
import torch
import math

# Triton kernel for row-wise dequantization
@triton.jit
def _dequantize_rowwise(
    x_ptr,           # Pointer to input tensor
    state_x,         # Pointer to row-wise state tensor (max values)
    output_ptr,      # Pointer to output tensor
    inv_127,         # Precomputed inverse of 127
    n_elements,      # Total number of elements in the tensor
    BLOCK_SIZE: tl.constexpr,  # Block size
    P2: tl.constexpr          # Nearest power of 2 of the number of columns
):
    # Program ID in axis 0 (row ID)
    pid = tl.program_id(axis=0)
    
    # Starting index of the block for the current row
    block_start = pid * n_elements
    
    # Create an index range [0, P2)
    arange = tl.arange(0, P2)
    
    # Compute global memory offsets for loading the input tensor
    offsets = block_start + arange
    
    # Create a mask to ensure out-of-bound accesses are ignored
    mask = arange < n_elements
    
    # Load input tensor values for the current row
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Load the maximum value for the current row from state_x
    max_val = tl.load(state_x + pid)
    
    # Dequantize the values
    dequantized = x * max_val * inv_127
    
    # Store the results back to the output tensor
    tl.store(output_ptr + offsets, dequantized, mask=mask)


# Python wrapper function for row-wise dequantization
def dequantize_rowwise(x, state_x, block_size):
    """
    Row-wise dequantization of the input tensor `x`.

    Args:
        x (torch.Tensor): Input tensor of shape (num_rows, num_columns).
        state_x (torch.Tensor): Tensor containing the row-wise maximum values.
        block_size (int): Block size for Triton kernel execution.

    Returns:
        torch.Tensor: Dequantized output tensor.
    """
    # Ensure input tensor is on CUDA
    assert x.is_cuda, "Input tensor `x` must be on CUDA."
    assert state_x.is_cuda, "State tensor `state_x` must be on CUDA."

    # Get the number of rows and columns
    num_rows, num_columns = x.shape

    # Precompute inverse of 127
    inv_127 = 1.0 / 127.0

    # Compute the nearest power of 2 of the number of columns
    P2 = 2 ** math.ceil(math.log2(num_columns))

    # Allocate output tensor
    output = torch.empty_like(x, device=x.device)

    # Grid size: One program per row
    grid = (num_rows,)

    # Launch the Triton kernel
    _dequantize_rowwise[grid](
        x_ptr=x,                    # Pointer to input tensor
        state_x=state_x,            # Pointer to row-wise state tensor
        output_ptr=output,          # Pointer to output tensor
        inv_127=inv_127,            # Precomputed inverse of 127
        n_elements=num_columns,     # Number of columns (elements per row)
        BLOCK_SIZE=block_size,      # Block size
        P2=P2                       # Nearest power of 2 of num_columns
    )

    return output

# Example usage
num_rows = 4
num_columns = 7
block_size = 16

# Input tensor (quantized) and row-wise max values
x = torch.randint(0, 128, (num_rows, num_columns), dtype=torch.float32, device='cuda')
state_x = torch.rand(num_rows, dtype=torch.float32, device='cuda')

# Dequantize the tensor
output = dequantize_rowwise(x, state_x, block_size)

print("Input Tensor:")
print(x)
print("Row-wise Max Values:")
print(state_x)
print("Dequantized Output Tensor:")
print(output)
