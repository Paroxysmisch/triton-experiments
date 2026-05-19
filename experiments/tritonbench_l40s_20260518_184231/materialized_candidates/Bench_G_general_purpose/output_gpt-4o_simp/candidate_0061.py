import triton
import triton.language as tl

@triton.jit
def _sgmv_expand_slice_kernel(
    # Pointers to data
    data_ptr, indices_ptr, indptr_ptr, lora_weights_ptr, output_ptr,
    # Sizes
    batch_size, num_rows, num_cols,
    # LoRA configuration
    lora_rank,
    # Block sizes
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    # Program ID for parallel execution
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute the start of the block for this program
    row_start = pid_m * BLOCK_SIZE_M
    col_start = pid_n * BLOCK_SIZE_N

    # Loop over the batch dimension
    for batch in range(batch_size):
        # Loop over rows in the block
        for row in range(row_start, min(row_start + BLOCK_SIZE_M, num_rows)):
            # Initialize the output for this row
            output_val = 0.0

            # Get the range of non-zero elements for this row
            start_idx = tl.load(indptr_ptr + row)
            end_idx = tl.load(indptr_ptr + row + 1)

            # Loop over non-zero elements in this row
            for idx in range(start_idx, end_idx):
                col = tl.load(indices_ptr + idx)
                if col_start <= col < col_start + BLOCK_SIZE_N:
                    # Load the value and the corresponding vector element
                    value = tl.load(data_ptr + idx)
                    vec_val = tl.load(lora_weights_ptr + batch * num_cols + col)

                    # Perform the multiplication and accumulate
                    output_val += value * vec_val

            # Store the result in the output tensor
            tl.store(output_ptr + batch * num_rows + row, output_val)

### Wrapper Function

def _sgmv_expand_slice(data, indices, indptr, lora_weights, batch_size, num_rows, num_cols, lora_rank):
    # Allocate output tensor
    output = torch.zeros((batch_size, num_rows), device=data.device, dtype=data.dtype)

    # Define block sizes
    BLOCK_SIZE_M = 128  # Adjust based on hardware capabilities
    BLOCK_SIZE_N = 128  # Adjust based on hardware capabilities

    # Launch the Triton kernel
    grid = (triton.cdiv(num_rows, BLOCK_SIZE_M), triton.cdiv(num_cols, BLOCK_SIZE_N))
    _sgmv_expand_slice_kernel[grid](
        data_ptr=data,
        indices_ptr=indices,
        indptr_ptr=indptr,
        lora_weights_ptr=lora_weights,
        output_ptr=output,
        batch_size=batch_size,
        num_rows=num_rows,
        num_cols=num_cols,
        lora_rank=lora_rank,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N
    )

    return output
