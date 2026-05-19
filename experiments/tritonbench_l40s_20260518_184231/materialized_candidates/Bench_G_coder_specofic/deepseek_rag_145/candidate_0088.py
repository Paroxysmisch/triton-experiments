import triton
import triton.language as tl

@triton.jit
def reduction_kernel(
    input_ptr,               # Pointer to input matrix
    output_ptr,               # Pointer to output matrix
    num_rows,                 # Number of rows in input matrix
    num_columns,               # Number of columns in input matrix
    reduction_op,             # Reduction operation (add, max, or min)
    BLOCK_DIM: tl.constexpr   # Block size
):
    # Define the columns block index.
    column_block_idx = tl.program_id(axis=0)

    # Define offsets for this block.
    offs = column_block_idx * BLOCK_DIM + tl.arange(0, BLOCK_DIM)

    # Load the block of data into shared memory.
    xs = tl.load(input_ptr + offs, mask=offs < num_columns)

    # Perform reduction operation in shared memory.
    ys = reduction_op(xs)

    # Store the results back to global memory.
    tl.store(output_ptr + column_block_idx, ys)


def reduction(input, output, reduction_op=tl.reduction.sum, num_streams=1):
    # Determine the size of the grid.
    grid = lambda meta: (triton.ceildiv(input.numel(), meta['BLOCK_DIM']), )

    # Perform reduction operation.
    reduction_kernel[grid](
        input.data_ptr(),
        output.data_ptr(),
        input.shape[0],
        input.shape[1],
        reduction_op,
    )
