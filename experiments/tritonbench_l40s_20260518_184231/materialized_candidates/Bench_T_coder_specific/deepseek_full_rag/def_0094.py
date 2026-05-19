import torch
import triton
import triton.language as tl
from functools import partial

class DropoutSigmoidLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None, p=0.5, training=True, inplace=False):
        if inplace:
            raise ValueError("Inplace mode is not supported for this function")

        output = torch.empty_like(input)
        assert input.is_contiguous()
        assert weight.is_contiguous()
        if bias is not None:
            assert bias.is_contiguous()

        size = input.numel()
        BLOCK_SIZE = 512
        grid = lambda meta: (triton.cdiv(size, meta['BLOCK_SIZE']),)
        dropout_forward_kernel[grid](
            input, output, size,
            p, 0,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        if training:
            ctx.p = p
            ctx.training = training
            ctx.save_for_backward(output)

        input = input.matmul(weight)
        if bias is not None:
            input += bias
        input = input.sigmoid()
        input *= output
        return input

    @staticmethod
    def backward(ctx, output_grad):
        if ctx.training:
            training = ctx.training
            p = ctx.p
            output, = ctx.saved_tensors
            assert output_grad.is_contiguous()
            assert output.is_contiguous()

            size = output.numel()
            BLOCK_SIZE = 512
            grid = lambda meta: (triton.cdiv(size, meta['BLOCK_SIZE']),)
            dropout_backward_kernel[grid](
                output_grad, output, size,
                p, 0,
                BLOCK_SIZE=BLOCK_SIZE,
            )
            output_grad = output * output_grad
        else:
            output_grad = output_grad.sigmoid()

        input_grad = output_grad.mm(ctx.weight)
        bias_grad = output_grad.sum(axis=0) if ctx.bias is not None else None

        ctx.weight.grad = output_grad.t().mm(ctx.input)
        if bias_grad is not None:
            ctx.bias.grad = bias_grad

        return input_grad, ctx.weight, bias_grad, None, None, None

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)

def dropout_sigmoid_linear(input: torch.Tensor, weight: torch.Tensor, bias=None, p=0.5, training=True, inplace=False) -> torch.Tensor:
    """
    Applies a linear transformation followed by a sigmoid activation and dropout.
    This function sequentially applies a linear transformation to the input tensor,
    a sigmoid activation to scale the values between 0 and 1, and randomly zeroes
    some elements of the tensor with a specified probability during dropout.

    Args:
        input (torch.Tensor): Input tensor of shape
            :math:`(*, \text{in\_features})`.
        weight (torch.Tensor): Weight tensor of shape
            :math:`(\text{out\_features}, \text{in\_features})`.
        bias (torch.Tensor): Bias tensor of shape
            :math:`(\text{out\_features})`. Default is `None`.
        p (float): Probability of an element to be zeroed in dropout.
            Default: 0.5.
        training (bool): If `True`, applies dropout during training.
            Default: `True`.
        inplace (bool): If `True`, performs the operation in-place.
            Default: `False`.

    Returns:
        torch.Tensor: The result of the linear transformation, followed by
            a sigmoid activation, and then dropout.
    """
    func = DropoutSigmoidLinearFunction.apply
    return func(input, weight, bias, p, training, inplace)
