import triton
import triton.language as tl

@triton.jit
def _dequantize_rowwise(
    x_ptr,  # Pointer to the input tensor
    state_x,  # Pointer to the state tensor (max values for each row)
    output_ptr,  # Pointer to the output tensor
    inv_127,  # Precomputed inverse of 127
    n_elements,  # Total number of elements in the input tensor
    BLOCK_SIZE: tl.constexpr,  # Block size
    P2: tl.constexpr  # Nearest power of two of the number of columns
):
    pid = tl.program_id(axis=0)  # Program ID in axis 0 (row ID)
    row_start = pid * P2  # Starting index for the current row

    # Calculate the starting index for the current block
    block_start = row_start + tl.arange(0, BLOCK_SIZE)

    # Create a mask to handle out-of-bound accesses
    mask = block_start < row_start + n_elements

    # Load the input values with the mask
    x = tl.load(x_ptr + block_start, mask=mask)

    # Load the maximum value for the current row
    max_val = tl.load(state_x + pid)

    # Dequantize the values
    output = x * max_val * inv_127

    # Store the results back to the output tensor
    tl.store(output_ptr + block_start, output, mask=mask)

import torch
import triton
import triton.language as tl

def dequantize_rowwise(x: torch.Tensor, state_x: torch.Tensor):
    # Ensure the input tensor is on CUDA
    assert x.is_cuda, "Input tensor must be on CUDA"
    assert state_x.is_cuda, "State tensor must be on CUDA"

    # Get the number of elements and the number of rows
    n_elements = x.shape[1]
    n_rows = x.shape[0]

    # Compute the nearest power of two of the number of columns
    P2 = 1 << (n_elements - 1).bit_length()

    # Prepare the output tensor
    output = torch.empty_like(x, device=x.device)

    # Precompute the inverse of 127
    inv_127 = 1.0 / 127.0

    # Set up the execution grid
    grid = (n_rows,)

    # Invoke the Triton kernel
    _dequantize_rowwise[grid](
        x,  # Pointer to the input tensor
        state_x,  # Pointer to the state tensor
        output,  # Pointer to the output tensor
        inv_127,  # Precomputed inverse of 127
        n_elements,  # Total number of elements in the input tensor
        BLOCK_SIZE=128,  # Block size
        P2=P2  # Nearest power of two of the number of columns
    )

    return output
