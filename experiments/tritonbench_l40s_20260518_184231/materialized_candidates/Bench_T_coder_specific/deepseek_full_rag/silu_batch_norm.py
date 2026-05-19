import torch
import triton
import triton.language as tl
from torch import Tensor

@triton.jit
def silu_batch_norm_kernel(
    input_ptr,
    running_mean_ptr,
    running_var_ptr,
    output_ptr,
    weight_ptr,
    bias_ptr,
    input_row_stride,
    output_row_stride,
    n_channels,
    n_rows,
    eps,
    momentum,
    training: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    row_id = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE_N)
    mask = col_offsets < n_channels

    running_mean = tl.load(running_mean_ptr)
    running_var = tl.load(running_var_ptr)
    inv_sqrt_var = tl.math.rsqrt(running_var + eps)

    row_start_ptr = input_ptr + row_id * input_row_stride + col_offsets
    x = tl.load(row_start_ptr, mask=mask, other=0.0).to(tl.float32)

    x_hat = (x - running_mean) * inv_sqrt_var

    weight = tl.load(weight_ptr + col_offsets, mask=mask)
    output = x_hat * weight

    if bias_ptr is not None:
        bias = tl.load(bias_ptr + col_offsets, mask=mask)
        output = output + bias

    output = output * (1.0 / (1.0 + tl.exp(-output)))

    output_row_start_ptr = output_ptr + row_id * output_row_stride + col_offsets
    tl.store(output_row_start_ptr, output, mask=mask)


class SiluBatchNorm(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        running_mean,
        running_var,
        weight,
        bias,
        training,
        momentum,
        eps,
    ):
        n_rows, n_channels = x.shape

        output = torch.empty_like(x)

        BLOCK_SIZE_N = triton.next_power_of_2(n_channels)
        grid = (n_rows,)

        silu_batch_norm_kernel[grid](
            x,
            running_mean,
            running_var,
            output,
            weight,
            bias,
            x.stride(0),
            output.stride(0),
            n_channels,
            n_rows,
            eps,
            momentum,
            training,
            BLOCK_SIZE_N,
        )

        ctx.save_for_backward(running_mean, running_var, weight, bias, x)
        ctx.training = training
        ctx.momentum = momentum
        ctx.eps = eps

        return output

    @staticmethod
    def backward(ctx, grad_output):
        (
            running_mean,
            running_var,
            weight,
            bias,
            input,
        ) = ctx.saved_tensors
        training = ctx.training
        momentum = ctx.momentum
        eps = ctx.eps

        n_rows, n_channels = input.shape

        grad_input = torch.empty_like(input)

        BLOCK_SIZE_N = triton.next_power_of_2(n_channels)
        grid = (n_rows,)

        silu_batch_norm_kernel[grid](
            grad_output,
            running_mean,
            running_var,
            grad_input,
            weight,
            bias,
            grad_output.stride(0),
            grad_input.stride(0),
            n_channels,
            n_rows,
            eps,
            momentum,
            training,
            BLOCK_SIZE_N,
        )

        grad_weight = torch.sum(grad_output * (input - running_mean) * weight, dim=0)
        grad_weight = grad_weight * (1.0 / n_rows)

        if bias is not None:
            grad_bias = torch.sum(grad_output * weight, dim=0)
            grad_bias = grad_bias * (1.0 / n_rows)
        else:
            grad_bias = None

        return grad_input, None, None, grad_weight, grad_bias, None, None, None


def silu_batch_norm(
    input, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-5
) -> Tensor:
    return SiluBatchNorm.apply(input, running_mean, running_var, weight, bias, training, momentum, eps)
