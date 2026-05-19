import torch
import triton
import triton.language as tl
from .utils import get_kernel_meta

@triton.jit
def fused_add_mul_activation_kernel(
    x_ptr, x_row_stride, x_col_stride,
    in_ptr, in_row_stride, in_col_stride,
    bias_ptr, bias_row_stride, bias_col_stride,
    out_ptr, out_row_stride, out_col_stride,
    n_rows, n_cols,
    multiplier: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step, BLOCK_SIZE):
        col_offsets = tl.arange(0, BLOCK_SIZE)
        col_mask = col_offsets < n_cols

        x_row_ptr = x_ptr + row_idx * x_row_stride
        in_row_ptr = in_ptr + row_idx * in_row_stride

        bias_row_ptr = bias_ptr + row_idx * bias_row_stride

        out_row_ptr = out_ptr + row_idx * out_row_stride

        x_row = tl.load(x_row_ptr + col_offsets, mask=col_mask, other=0.0)
        in_row = tl.load(in_row_ptr + col_offsets, mask=col_mask, other=0.0)
        bias = tl.load(bias_row_ptr + col_offsets, mask=col_mask, other=0.0)

        x_scaled = x_row * multiplier
        tmp = x_scaled + in_row * bias
        if ACTIVATION == "sigmoid":
            out = tl.sigmoid(tmp)
        elif ACTIVATION == "relu":
            out = tl.where(tmp >= 0, tmp, 0.0)
        tl.store(out_row_ptr + col_offsets, out, mask=col_mask)

def fused_add_mul_activation_torch(
    x: torch.Tensor,
    in_out_tensor: torch.Tensor,
    bias: torch.Tensor,
    multiplier: float = 1.0,
    activation: str = "sigmoid",
) -> torch.Tensor:
    assert x.shape[-1] == bias.shape[-1]
    assert x.is_contiguous()

    out = in_out_tensor if in_out_tensor is not None else torch.empty_like(x)
    assert out.shape == x.shape
    assert out.is_contiguous()

    n_rows, n_cols = x.numel() // x.shape[-1], x.shape[-1]
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    num_warps = 4
    if BLOCK_SIZE > 2047:
        num_warps = 8
    elif BLOCK_SIZE > 4095:
        num_warps = 16

    grid = (triton.cdiv(n_rows, BLOCK_SIZE), 1, 1)
    kernel_meta = get_kernel_meta(x)
    fused_add_mul_activation_kernel[grid](
        x, *x.stride(),
        in_out_tensor, *in_out_tensor.stride(),
        bias, *bias.stride(),
        out, *out.stride(),
        n_rows, n_cols,
        multiplier=multiplier,
        BLOCK_SIZE=BLOCK_SIZE,
        ACTIVATION=activation,
        num_warps=num_warps,
        num_stages=1,
        **kernel_meta,
    )
    return out
