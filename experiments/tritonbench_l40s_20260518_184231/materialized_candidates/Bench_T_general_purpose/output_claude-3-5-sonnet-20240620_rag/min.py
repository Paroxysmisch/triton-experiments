import torch
import triton
import triton.language as tl

@triton.jit
def min_kernel(output_ptr, indices_ptr, input_ptr, input_row_stride, output_row_stride, n_cols, dim, keepdim, BLOCK_SIZE: tl.constexpr):
    # The rows are independent, so we parallelize across those
    row_idx = tl.program_id(0)
    # The stride represents how much we need to increase the pointer to advance 1 row
    row_start_ptr = input_ptr + row_idx * input_row_stride
    # The block size is the next power of two greater than n_cols, so we can fit each
    # row in a single block
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets

    # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=float('inf'))

    # Initialize min value and index
    min_value = float('inf')
    min_index = -1

    # Find the minimum value and its index
    for i in range(n_cols):
        if row[i] < min_value:
            min_value = row[i]
            min_index = i

    # Write back output to DRAM
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    indices_row_start_ptr = indices_ptr + row_idx * output_row_stride

    if keepdim:
        tl.store(output_row_start_ptr, min_value)
        tl.store(indices_row_start_ptr, min_index)
    else:
        tl.store(output_row_start_ptr, min_value)
        tl.store(indices_row_start_ptr, min_index)

def min(input: torch.Tensor, dim: int, keepdim: bool = False, *, out: tuple = None) -> (torch.Tensor, torch.Tensor):
    n_rows, n_cols = input.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    # Allocate output tensors
    min_values = torch.empty((n_rows,), device=input.device, dtype=input.dtype)
    min_indices = torch.empty((n_rows,), device=input.device, dtype=torch.long)

    min_kernel[(n_rows,)](
        min_values,
        min_indices,
        input,
        input.stride(0),
        min_values.stride(0),
        n_cols,
        dim,
        keepdim,
        BLOCK_SIZE=BLOCK_SIZE
    )

    if keepdim:
        return min_values.view(n_rows, 1), min_indices.view(n_rows, 1)
    else:
        return min_values, min_indices
