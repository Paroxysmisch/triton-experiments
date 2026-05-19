import triton
import triton.language as tl
import torch

@triton.jit
def softmax_kernel(input_ptr, output_ptr, input_row_stride, output_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    # Program ID corresponds to the row index
    row_idx = tl.program_id(axis=0)
    
    # Compute memory offsets for input and output
    input_offset = row_idx * input_row_stride
    output_offset = row_idx * output_row_stride
    
    # Create a range of column indices
    col_idx = tl.arange(0, BLOCK_SIZE)
    
    # Compute the memory location for this block
    input_ptrs = input_ptr + input_offset + col_idx
    output_ptrs = output_ptr + output_offset + col_idx
    
    # Mask to ensure we don't read/write out of bounds
    mask = col_idx < n_cols
    
    # Load data from input with masking, filling absent values with -inf
    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
    
    # Compute the maximum value in the row for numerical stability
    row_max = tl.max(row, axis=0)
    
    # Subtract the max, exponentiate, and compute the sum
    row = row - row_max
    numerator = tl.exp(row)
    denominator = tl.sum(numerator, axis=0)
    
    # Compute softmax
    softmax_result = numerator / denominator
    
    # Store the result back to output
    tl.store(output_ptrs, softmax_result, mask=mask)

def triton_softmax(x):
    # Extract dimensions
    n_rows, n_cols = x.shape
    
    # Initialize output tensor
    output = torch.empty_like(x)
    
    # Choose BLOCK_SIZE as the next power of two greater than or equal to n_cols, capped at 1024
    BLOCK_SIZE = min(2**((n_cols - 1).bit_length()), 1024)
    
    # Configure grid size to cover all rows
    grid = (n_rows,)
    
    # Launch the kernel
    softmax_kernel[grid](x, output, x.stride(0), output.stride(0), n_cols, BLOCK_SIZE=BLOCK_SIZE)
    
    return output

# Example usage:
x = torch.randn(128, 512, device='cuda')  # Input tensor
output = triton_softmax(x)  # Compute softmax
