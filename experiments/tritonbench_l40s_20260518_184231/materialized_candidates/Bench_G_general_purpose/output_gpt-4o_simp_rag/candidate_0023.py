import triton
import triton.language as tl
import torch

@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    # Get the row index this kernel instance is responsible for
    row_idx = tl.program_id(0)
    # Compute the starting pointer of the row in the input matrix
    row_start_ptr = input_ptr + row_idx * input_row_stride
    # Compute column offsets for the current block
    col_offsets = tl.arange(0, BLOCK_SIZE)
    # Compute pointers for the current row and block
    input_ptrs = row_start_ptr + col_offsets
    # Load the row into SRAM with a mask for valid columns
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float("inf"))
    # Convert to float32 for numerical stability
    row_f32 = row.to(tl.float32)
    # Subtract the maximum value for numerical stability
    row_minus_max = row_f32 - tl.max(row_f32, axis=0)
    # Compute the exponentials
    numerator = tl.exp(row_minus_max)
    # Compute the sum of the exponentials
    denominator = tl.sum(numerator, axis=0)
    # Compute the softmax output
    softmax_output = numerator / denominator
    # Compute the starting pointer of the row in the output matrix
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    # Compute pointers for the current row and block in the output matrix
    output_ptrs = output_row_start_ptr + col_offsets
    # Store the softmax result back to DRAM
    tl.store(output_ptrs, softmax_output.to(row.dtype), mask=col_offsets < n_cols)

def softmax(input_tensor):
    # Get the shape of the input tensor
    n_rows, n_cols = input_tensor.shape
    # Determine the block size (next power of two greater than n_cols)
    BLOCK_SIZE = 1 << (n_cols - 1).bit_length()
    # Create an output tensor
    output_tensor = torch.empty_like(input_tensor)
    # Determine the number of warps based on the block size
    num_warps = 4 if BLOCK_SIZE < 2048 else 8 if BLOCK_SIZE < 4096 else 16
    # Launch the kernel
    grid = (n_rows,)
    softmax_kernel[grid](output_tensor, input_tensor, input_tensor.stride(0), output_tensor.stride(0), n_cols, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)
    return output_tensor

# Example usage
input_matrix = torch.randn(128, 512, device='cuda', dtype=torch.float32)
output_matrix = softmax(input_matrix)
