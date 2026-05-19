import torch
import triton
import triton.language as tl
from torch import Tensor
from triton.runtime import runtime

@triton.jit
def log_softmax_linear_kernel(input, weight, bias, dim, dtype,
                               stride_input_n, stride_input_f,
                               stride_weight_n, stride_weight_f,
                               stride_bias_n,
                               BLOCK_N: tl.constexpr, BLOCK_F: tl.constexpr):
    # Triton kernel implementation
    # Calculate linear transformation and log_softmax
    pid_n = tl.program_id(0)
    pid_f = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_f = pid_f * BLOCK_F + tl.arange(0, BLOCK_F)

    mask_n = offs_n < input.shape[0]
    mask_f = offs_f < input.shape[1]

    input_ptrs = input + offs_n * stride_input_n + offs_f * stride_input_f
    weight_ptrs = weight + offs_f * stride_weight_f
    bias_ptrs = bias + offs_n * stride_bias_n

    linear_input = tl.load(input_ptrs, mask=mask_n[:, None] & mask_f[None, :])
    linear_weight = tl.load(weight_ptrs, mask=mask_f[:])
    linear_bias = tl.load(bias_ptrs, mask=mask_n[:])

    linear_output = tl.dot(linear_input, linear_weight) + linear_bias

    if dtype is not None:
        linear_output = linear_output.to(dtype)

    logits_max, _ = tl.max(linear_output, axis=1, return_indices=True)
    logits_shifted = linear_output - logits_max[:, None]
    exp_logits = tl.exp(logits_shifted)
    sum_exp_logits, _ = tl.sum(exp_logits, axis=1, return_indices=True)
    log_softmax_output = logits_shifted - tl.log(sum_exp_logits)[:, None]

    return log_softmax_output

class LogSoftmaxLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None, dim=-1, dtype=None):
        # Forward pass logic
        input = input.contiguous()
        weight = weight.contiguous()
        if bias is not None:
            bias = bias.contiguous()

        if dim != -1:
            input = input.transpose(0, dim)

        input_shape = input.shape
        feature_size = input_shape[-1]
        batch_size = int(torch.tensor(input_shape[:-1]).prod().item())
        num_features = weight.shape[0]

        if dtype is None:
            dtype = input.dtype

        output_shape = list(input_shape[:-1]) + [num_features]
        output = torch.empty(output_shape, dtype=dtype, device=input.device)

        grid = lambda meta: (
            triton.cdiv(batch_size, meta['BLOCK_N']),
            triton.cdiv(feature_size, meta['BLOCK_F'])
        )

        log_softmax_linear_kernel[grid](
            input, weight, bias, dim, dtype,
            input.stride(0), input.stride(1),
            weight.stride(0), weight.stride(1),
            output.stride(0) if bias is None else bias.stride(0),
        )

        if dim != -1:
            output = output.transpose(0, dim)

        ctx.save_for_backward(output)
        ctx.dim = dim

        return output

    @staticmethod
    def backward(ctx, output_grad):
        # Backward pass logic
        output, = ctx.saved_tensors

        if ctx.dim != -1:
            output_grad = output_grad.transpose(0, ctx.dim)

        output_grad = output_grad.contiguous()
        input_grad = torch.empty_like(output)

        grid = lambda meta: (
            triton.cdiv(output.shape[0], meta['BLOCK_N']),
            triton.cdiv(output.shape[1], meta['BLOCK_F'])
        )

        runtime.reset_and_jit(
            prune_reset_to_zero=["input_grad", "output"],
            reset_to_zero=["input_grad"]
        )

        log_softmax_linear_kernel[grid](
            output, weight, bias, dim, dtype,
            output.stride(0), output.stride(1),
            weight.stride(0), weight.stride(1),
            input_grad.stride(0) if bias is None else bias.stride(0),
        )

        input_grad = input_grad * output_grad

        if ctx.dim != -1:
            output_grad = output_grad.transpose(0, ctx.dim)

        return input_grad, None, None, None, None

def log_softmax_linear(input, weight, bias=None, dim=-1, dtype=None):
    # Wrapper function for Triton kernel
    return LogSoftmaxLinearFunction.apply(input, weight, bias, dim, dtype)
