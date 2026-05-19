import torch
import triton
import triton.language as tl
from typing import Optional

@triton.jit
def argmax_kernel(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float('inf'))
    indices = col_offsets
    # Create pairs of (value, index)
    pairs = tl.stack([row, indices], axis=1)
    # Define the combine function
    @tl.jit
    def combine(a, b):
        a_val, a_idx = a[0], a[1]
        b_val, b_idx = b[0], b[1]
        # Choose the pair with higher value, or lower index if equal
        condition = (a_val > b_val) | ((a_val == b_val) & (a_idx < b_idx))
        return tl.where(condition, a, b)
    # Reduce across the row to find the max pair
    max_pair = tl.reduce(pairs, 0, combine)
    max_idx = max_pair[1]
    # Write to output
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    tl.store(output_row_start_ptr, max_idx.to(tl.int64))

def argmax(input: torch.Tensor, dim: Optional[int] = None, keepdim: bool = False) -> torch.LongTensor:
    if dim is None:
        input_flat = input.flatten()
        n_cols = input_flat.shape[0]
        if n_cols == 0:
            return torch.tensor(-1, dtype=torch.int64, device=input.device)
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        output = torch.empty((1,), dtype=torch.int64, device=input.device)
        argmax_kernel[(1,)](output, input_flat, input_flat.stride(0), output.stride(0), n_cols, BLOCK_SIZE=BLOCK_SIZE)
        return output.view(())
    else:
        if dim < 0:
            dim += input.dim()
        input_transposed = input.transpose(dim, -1).contiguous()
        input_2d = input_transposed.flatten(0, -2)
        n_rows, n_cols = input_2d.shape
        if n_cols == 0:
            return torch.zeros(input_2d.shape[0], dtype=torch.int64, device=input.device).view(
                *(input_transposed.shape[:-1] + (1,) if keepdim else input_transposed.shape[:-1])
            ).transpose(dim, -1).contiguous()
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        output_shape = (n_rows, 1) if keepdim else (n_rows,)
        output = torch.empty(output_shape, dtype=torch.int64, device=input.device)
        grid = (n_rows,)
        argmax_kernel[grid](
            output, input_2d,
            input_2d.stride(0),
            output.stride(0),
            n_cols,
            BLOCK_SIZE=BLOCK_SIZE
        )
        original_transposed_shape = input_transposed.shape
        if keepdim:
            new_shape = list(original_transposed_shape[:-1]) + [1]
        else:
            new_shape = list(original_transposed_shape[:-1])
        output_reshaped = output.reshape(new_shape)
        perm = list(range(input_transposed.dim()))
        perm[dim], perm[-1] = perm[-1], perm[dim]
        output_transposed = output_reshaped.permute(perm)
        return output_transposed.contiguous()
