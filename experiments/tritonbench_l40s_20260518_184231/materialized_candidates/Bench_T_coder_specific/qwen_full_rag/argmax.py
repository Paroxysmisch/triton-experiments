import torch
import triton
import triton.language as tl


@triton.jit
def argmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    # The rows are independent, so we parallelize across those
    row_idx = tl.program_id(0)
    # The stride represents how much we need to increase the pointer to advance 1 row
    row_start_ptr = input_ptr + row_idx * input_row_stride
    # The block size is the next power of two greater than n_cols, so we can fit each
    # row in a single block
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=float('-inf'))
    max_value = tl.max(row, axis=0)
    max_indices = tl.where(row == max_value, col_offsets, -1)
    selected_index = tl.argmax(row, axis=0)
    result = tl.where(col_offsets == selected_index, selected_index, -1)
    
    # Write back output to DRAM
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, result, mask=col_offsets < n_cols)


def argmax(input, dim=None, keepdim=False):
    """
    See :func:`torch.argmax`
    """

    assert dim is not None, "Only dim is supported"

    input = input.contiguous()

    if not keepdim:
        input = tl.drop_dim(input, dim)
    
    n_dims = len(input.shape)
    n_cols = 1
    for i in range(n_dims - 1):
        n_cols *= input.shape[i]

    BLOCK_SIZE = triton.next_power_of_2(input.shape[-1])

    # Allocate output
    output = torch.empty(n_cols, device=input.device, dtype=torch.int64)
    
    grid = lambda meta: (triton.cdiv(n_cols, meta["BLOCK_SIZE"]), )

    argmax_kernel[grid](output, input, input.stride(0), output.stride(0), n_cols, BLOCK_SIZE=BLOCK_SIZE)
    
    if not keepdim:
        output = output.unsqueeze(dim)
    
    return output
