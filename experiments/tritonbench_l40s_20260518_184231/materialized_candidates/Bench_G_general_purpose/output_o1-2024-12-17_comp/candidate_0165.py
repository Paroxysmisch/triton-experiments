import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
    output_ptr, input_ptr, row_stride, n_cols, mask_ptr, BLOCK_SIZE: tl.constexpr
):
    row_id = tl.program_id(0)
    row_offset = row_id * row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    in_ptrs = input_ptr + row_offset + col_offsets
    x = tl.load(in_ptrs, mask=mask, other=-float('inf'))

    if mask_ptr != 0:
        mask_vals = tl.load(mask_ptr + row_offset + col_offsets, mask=mask, other=0.0)
        x = x + mask_vals

    x_max = tl.maximum(tl.max(x, mask=mask), 0.0)
    x = x - x_max
    x_exp = tl.exp(x)
    s = tl.sum(x_exp, mask=mask)
    inv_s = 1.0 / s
    out = x_exp * inv_s

    out_ptrs = output_ptr + row_offset + col_offsets
    tl.store(out_ptrs, out, mask=mask)


def softmax(input: torch.Tensor, mask: torch.Tensor = None, dim: int = -1) -> torch.Tensor:
    if dim != -1:
        raise ValueError("Only dim=-1 is supported.")
    if input.dim() > 2:
        input_2d = input.view(-1, input.shape[-1])
    else:
        input_2d = input
    n_rows, n_cols = input_2d.shape
    if mask is not None and mask.shape != input_2d.shape:
        raise ValueError("Mask shape must match input shape along the last dimension.")
    output = torch.empty_like(input_2d)

    BLOCK_SIZE = 128
    num_warps = 4 if n_cols > 1024 else 1
    grid = (n_rows,)

    mask_ptr = mask.data_ptr() if mask is not None else 0

    softmax_kernel[grid](
        output, input_2d, input_2d.stride(0), n_cols, mask_ptr,
        BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps, num_stages=2
    )

    if input.dim() > 2:
        return output.view(input.shape)
    return output
