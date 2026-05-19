import torch
import triton
import triton.language as tl

@triton.jit
def logsumexp_kernel(input, output, input_row_stride, output_row_stride, n_rows,
                     n_cols, BLOCK_SIZE: tl.constexpr):
    row_start = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    row_ptr = input + row_start * input_row_stride + col_offsets
    row = tl.load(row_ptr, mask=mask, other=-float('inf'))
    row_max = tl.max(row, axis=0)
    safe_row = tl.where(mask, row - row_max, -float('inf'))
    sum_exp_row = tl.sum(tl.exp(safe_row), axis=0)
    logsumexp = tl.log(sum_exp_row) + row_max

    output_row_ptr = output + row_start * output_row_stride
    tl.store(output_row_ptr, logsumexp)

def logsumexp(input, dim, keepdim=False, *, out=None):
    if out is None:
        out = torch.empty(input.shape, dtype=input.dtype, device=input.device)
    else:
        assert out.shape == input.shape and out.dtype == input.dtype and out.device == input.device

    if not keepdim:
        input = torch.squeeze(input, dim)
        out = torch.squeeze(out, dim)

    n_rows, n_cols = input.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    grid = (n_rows,)
    logsumexp_kernel[grid](input, out, input.stride(0), out.stride(0), n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE)

    if not keepdim:
        out = out.unsqueeze(dim=dim)

    return out
