import triton
import triton.language as tl

@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    # Obtain the row index for this program instance
    row_idx = tl.program_id(axis=0)
    
    # Compute the starting pointer for the row in input and output
    input_row_ptr = input_ptr + row_idx * input_row_stride
    output_row_ptr = output_ptr + row_idx * output_row_stride
    
    # Load the row into SRAM
    row = tl.load(input_row_ptr + tl.arange(0, n_cols), mask=tl.arange(0, n_cols) < n_cols)
    
    # Compute the maximum value of the row for numerical stability
    row_max = tl.max(row, axis=0)
    
    # Compute the numerator by subtracting the max and applying exp
    numerator = tl.exp(row - row_max)
    
    # Compute the denominator by summing the numerator
    denominator = tl.sum(numerator, axis=0)
    
    # Compute the softmax result by dividing the numerator by the denominator
    softmax_result = numerator / denominator
    
    # Store the result back to the output matrix
    tl.store(output_row_ptr + tl.arange(0, n_cols), softmax_result, mask=tl.arange(0, n_cols) < n_cols)

def triton_softmax(input_matrix, BLOCK_SIZE):
    # Determine the number of rows and columns in the input matrix
    n_rows, n_cols = input_matrix.shape
    
    # Allocate output matrix
    output_matrix = torch.empty_like(input_matrix)
    
    # Launch the Triton kernel for each row
    grid = (n_rows,)
    
    softmax_kernel[grid](
        output_matrix,
        input_matrix,
        input_matrix.stride(0),
        output_matrix.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output_matrix
