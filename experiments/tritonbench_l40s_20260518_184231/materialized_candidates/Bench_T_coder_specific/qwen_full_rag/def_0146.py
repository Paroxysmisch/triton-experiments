import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.autograd.function import Function
from .utils import get_tensor_size, contiguous


class EluLinearFunction(Function):

    @staticmethod
    def forward(ctx, input: Tensor, weight: Tensor, bias: Tensor = None,
                alpha: float = 1.0, inplace: bool = False) -> Tensor:
        input = contiguous(input)
        weight = contiguous(weight)
        if bias is not None:
            bias = contiguous(bias)
        
        output = torch.empty(get_tensor_size(input, 0), get_tensor_size(weight, 0),
                             dtype=input.dtype, device=input.device)
        n_elements = output.numel()
        block_size = triton.next_power_of_2(math.ceil(math.sqrt(n_elements)))
        grid = (triton.cdiv(n_elements, block_size), )
        
        EluLinearFunction.apply_kernel[grid](input, weight, bias, alpha, 
                                             output, n_elements, block_size, inplace)
        ctx.save_for_backward(input, weight, bias)
        ctx.alpha = alpha
        ctx.inplace = inplace
        
        return output
    
    @staticmethod
    @triton.jit
    def apply_kernel(x_ptr: tl.tensor, weight_ptr: tl.tensor, bias_ptr: tl.tensor,
                     alpha: float, output_ptr: tl.tensor, n_elements: int,
                     BLOCK_SIZE: tl.constexpr, inplace: bool):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        thread_offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = thread_offsets < n_elements
        x = tl.load(x_ptr + thread_offsets, mask=mask)
        weight = tl.load(weight_ptr + thread_offsets, mask=mask)
        if bias_ptr is not None:
            bias = tl.load(bias_ptr + thread_offsets, mask=mask)
        
        if inplace:
            output = x * weight + bias
            output = tl.where(x > 0, output, alpha * (tl.exp(output) - 1))
            tl.store(output_ptr + thread_offsets, output, mask=mask)
        else:
            output = x * weight + bias
            output = tl.where(x > 0, output, alpha * (tl.exp(output) - 1))
            tl.store(output_ptr + thread_offsets, output, mask=mask)


def elu_linear(input: Tensor, weight: Tensor, bias: Tensor = None,
               alpha: float = 1.0, inplace: bool = False) -> Tensor:
    r"""
    Apply elu to the result of linear transformation.

    Args:
        input (Tensor): the input tensor for linear layer.
        weight (Tensor): the weight tensor for linear transformation.
        bias (Tensor, optional): the bias tensor for linear transformation. Default: None.
        alpha (float, optional): the :math:`\alpha` parameter for ELU function. Default: 1.0.
        inplace (bool, optional): whether to apply ELU in-place. Default: False.

    Returns:
        Tensor: the output tensor.

    Note:
        The linear layer is applied first, then the ELU activation function is applied element-wise.
    """
    return EluLinearFunction.apply(input, weight, bias, alpha, inplace)
