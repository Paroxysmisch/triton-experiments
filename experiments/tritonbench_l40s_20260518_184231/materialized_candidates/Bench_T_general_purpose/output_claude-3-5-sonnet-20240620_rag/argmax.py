import torch
import triton
import triton.language as tl

@triton.jit
def argmax_kernel(output_ptr, input_ptr, input_row_stride, n_cols, dim, keepdim, BLOCK_SIZE: tl.constexpr):
    # The rows are independent, so we parallelize across those
    row_idx = tl.program_id(0)
    # The stride represents how much we need to increase the pointer to advance 1 row
    row_start_ptr = input_ptr + row_idx * input_row_stride

    # Initialize variables for tracking max values and indices
    max_value = -float('inf')
    max_index = -1

    if dim is None:
        # Flatten the input tensor
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float('inf'))

        # Find the maximum value and its index
        for i in range(n_cols):
            if row[i] > max_value:
                max_value = row[i]
                max_index = i

    else:
        # Handle the case where dim is specified
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float('inf'))

        # Find the maximum value and its index along the specified dimension
        for i in range(n_cols):
            if row[i] > max_value:
                max_value = row[i]
                max_index = i

    # Write back the index of the maximum value to the output tensor
    output_row_start_ptr = output_ptr + row_idx * (1 if keepdim else 0)
    tl.store(output_row_start_ptr, max_index)

def argmax(input: torch.Tensor, dim: int = None, keepdim: bool = False) -> torch.LongTensor:
    n_rows, n_cols = input.shape
    output_shape = (n_rows, 1) if keepdim else (n_rows,)
    output = torch.empty(output_shape, dtype=torch.long, device=input.device)

    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    argmax_kernel[(n_rows,)](
        output,
        input,
        input.stride(0),
        n_cols,
        dim,
        keepdim,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return output
