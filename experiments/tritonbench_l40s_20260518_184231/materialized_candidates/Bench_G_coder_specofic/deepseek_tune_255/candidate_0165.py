import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(output_ptr, input_ptr, row_stride, n_cols, mask_ptr, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float("inf"))
    row_minus_max = row - tl.max(row, axis=0)
    if mask_ptr is not None:
        mask = tl.load(mask_ptr + row_idx)
        row_minus_max = tl.where(mask, row_minus_max, -float("inf"))
    exp_row = tl.exp(row_minus_max)
    denom = tl.sum(exp_row, axis=0)
    softmax_output = exp_row / denom
    output_row_start_ptr = output_ptr + row_idx * row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=col_offsets < n_cols)

def softmax(input: torch.Tensor, mask: torch.Tensor = None, dim=-1) -> torch.Tensor:
    assert input.is_contiguous()
    assert dim in [-1, 1], "Only -1 and 1 are supported for dim"
    if mask is not None:
        assert mask.is_contiguous()
        assert mask.shape[0] == input.shape[0]
    if dim == -1:
        assert input.ndim >= 2
        if input.ndim > 2:
            input = input.reshape(-1, input.shape[-1])
        n_rows, n_cols = input.shape
    else:
        input = input.unsqueeze(0)
        n_rows, n_cols = input.shape[-2:]
        dim = -2
    output = torch.empty_like(input)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    num_warps = 4
    if BLOCK_SIZE > 2047:
        num_warps = 8
    if BLOCK_SIZE > 4095:
        num_warps = 16
    grid = lambda meta: (triton.cdiv(n_rows, meta["BLOCK_SIZE"]),)
    softmax_kernel[grid](
        output,
        input,
        input.stride(0),
        n_cols,
        mask,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return output
