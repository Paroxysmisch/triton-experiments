import logging
from typing import Optional

import torch
import triton
import triton.language as tl


try:
    from triton.language.extra.cuda.libdevice import log1pexp as _log1pexp
except ImportError:
    try:
        from triton.language.math import log1pexp as _log1pexp
    except ImportError:
        from triton.language.libdevice import log1pexp as _log1pexp


@triton.jit
def logsumexp_kernel(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_start = tl.program_id(axis=0)
    row_step = tl.num_programs(axis=0)
    for row_idx in tl.range(row_start, n_rows, row_step):
        col_offsets = tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        row_start_ptr = input_ptr + row_idx * input_row_stride + col_offsets
        row_data = tl.load(row_start_ptr, mask=mask, other=-float("inf"))
        row_max = tl.max(row_data, axis=0)
        row_exp_sum = tl.sum(_log1pexp(row_data.to(tl.float32)), axis=0)
        output_row_start_ptr = (
            output_ptr + row_idx * output_row_stride + col_offsets[:1]
        )
        tl.store(output_row_start_ptr, row_max + row_exp_sum, mask=mask[:1])


class LogSumExp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, dim: int, keepdim: bool = False) -> torch.Tensor:
        logging.debug("GEMS LOGSUMEXP FORWARD")

        if dim is None:
            x_flattened = x.contiguous().flatten()
            dim = -1
        else:
            x_flattened = x.contiguous()

        input_shape = x_flattened.shape
        x_flattened = x_flattened.reshape(-1, input_shape[-1])
        n_rows, n_cols = x_flattened.shape

        output_shape = list(input_shape)
        if dim < 0:
            dim = dim % x.ndim
        output_shape[dim] = 1

        if not keepdim:
            output_shape = [d for d, s in zip(output_shape, input_shape) if s != 1]

        block_size = triton.next_power_of_2(n_cols)
        grid = (1,)
        logsumexp_kernel[grid](
            x_flattened,
            x_flattened,
            x_flattened.stride(0),
            x_flattened.stride(1),
            n_rows,
            n_cols,
            BLOCK_SIZE=block_size,
        )

        output = torch.empty(output_shape, dtype=x.dtype, device=x.device)
        ctx.save_for_backward(x)
        ctx.dim = dim
        ctx.keepdim = keepdim
        return output.squeeze(dim=dim) if not keepdim else output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        logging.debug("GEMS LOGSUMEXP BACKWARD")
        (x,) = ctx.saved_tensors
        dim = ctx.dim
        keepdim = ctx.keepdim

        if dim is None:
            x_unflattened = x.reshape(-1)
            grad_output_unflattened = grad_output.unsqueeze(0).reshape(-1)
            dim = -1
        else:
            shape_before_dim = list(x.shape)[:dim]
            shape_after_dim = list(x.shape)[dim + 1 :]
            x_unflattened = x.reshape(-1, *shape_after_dim)
            grad_output_unflattened = grad_output.reshape(
                -1, *shape_after_dim
            ) if not keepdim else grad_output.reshape(*shape_before_dim, -1)
        n_rows, n_cols = x_unflattened.shape

        grad_input_shape = list(grad_output_unflattened.shape)[:-1] + [x_unflattened.shape[-1]]
        grad_input = torch.zeros(grad_input_shape, dtype=x.dtype, device=x.device)

        block_size = triton.next_power_of_2(n_cols)
        grid = (1,)
        logsumexp_kernel[grid](
            grad_input,
            x_unflattened,
            grad_input.stride(0),
            x_unflattened.stride(0),
            n_rows,
            n_cols,
            BLOCK_SIZE=block_size,
        )

        grad_input *= grad_output_unflattened
        if dim < 0:
            dim = dim % x.ndim
        grad_input = grad_input.reshape(*grad_output.shape, x.shape[dim])

        return grad_input, None, None


def logsumexp(
    input: torch.Tensor,
    dim: int,
    keepdim: bool = False,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    logging.debug("GEMS LOGSUMEXP")

    if out is not None:
        raise ValueError("GEMS DOES NOT SUPPORT PREALLOCATED OUTPUT TENSORS")

    return LogSumExp.apply(input, dim, keepdim)
