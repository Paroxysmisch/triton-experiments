import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
    input_ptr,  # Pointer to the input matrix
    output_ptr,  # Pointer to the output matrix
    input_row_stride,  # Stride between rows in the input matrix
    output_row_stride,  # Stride between rows in the output matrix
    n_cols,  # Number of columns in the matrix
    BLOCK_SIZE: tl.constexpr,  # Block size (power of two greater than n_cols)
):
    # Determine the row index for the current program
    row_idx = tl.program_id(0)
    
    # Compute the starting pointers for the current row
    row_start_ptr_input = input_ptr + row_idx * input_row_stride
    row_start_ptr_output = output_ptr + row_idx * output_row_stride
    
    # Generate column offsets within the block
    col_offsets = tl.arange(0, BLOCK_SIZE)
    
    # Create pointers and mask to handle potential out-of-bounds access
    input_ptrs = row_start_ptr_input + col_offsets
    mask = col_offsets < n_cols
    
    # Load the row data with masking (out-of-bounds elements are set to -infinity)
    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
    
    # Numerical stabilization: subtract maximum for numerical stability
    row_minus_max = row - tl.max(row, axis=0)
    
    # Compute exponentials and sum for normalization
    exp_values = tl.exp(row_minus_max)
    sum_exp = tl.sum(exp_values, axis=0)
    softmax_output = exp_values / sum_exp
    
    # Store the results back to memory with the same mask
    output_ptrs = row_start_ptr_output + col_offsets
    tl.store(output_ptrs, softmax_output, mask=mask)

def softmax(x: torch.Tensor) -> torch.Tensor:
    # Ensure the input tensor is 2D
    assert x.dim() == 2, "Input tensor must be 2-dimensional"
    
    M, N = x.shape
    if N == 0:
        raise ValueError("Number of columns must be positive")
    
    # Determine the optimal block size (smallest power of two >= N)
    BLOCK_SIZE = 1 << (N - 1).bit_length()
    
    # Configure number of warps based on block size
    num_warps = (BLOCK_SIZE + 31) // 32  # Ensures at least one warp
    
    # Initialize output tensor
    output = torch.empty_like(x)
    
    # Launch the Triton kernel with one block per row
    grid = (M,)
    softmax_kernel[grid](
        x.data_ptr(),
        output.data_ptr(),
        x.stride(0),
        output.stride(0),
        N,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps
    )
    
    return output

# Example usage
x = torch.randn(1024, 768, device='cuda')  # Input tensor on GPU
result = softmax(x)  # Compute softmax using Triton
print(result)
