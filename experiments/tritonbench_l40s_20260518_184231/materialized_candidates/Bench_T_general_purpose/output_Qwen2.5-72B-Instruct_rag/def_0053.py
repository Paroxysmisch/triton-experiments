import torch
import triton
import triton.language as tl
import torch.nn.functional as F

@triton.jit
def mul_relu_kernel(
    input_ptr,
    other_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    input_vals = tl.load(input_ptr + offsets, mask=mask)
    other_vals = tl.load(other_ptr + offsets, mask=mask)

    mul_vals = input_vals * other_vals
    relu_vals = tl.maximum(mul_vals, 0)

    tl.store(output_ptr + offsets, relu_vals, mask=mask)

import torch
import triton
from torch import Tensor

@torch.inference_mode()
def mul_relu(input: Tensor, other: Tensor, inplace: bool = False, out: Tensor = None) -> Tensor:
    """
    Performs element-wise multiplication of two inputs, `input` and `other`, and then applies the ReLU function to the result.
    
    Args:
        input (Tensor): The input tensor to be multiplied.
        other (Tensor or Number): The tensor or number to multiply with `input`.
        inplace (bool, optional): If True, modifies `input` in-place, if possible. Default is False.
        out (Tensor, optional): The output tensor.
    
    Returns:
        Tensor: The output tensor after applying the element-wise multiplication and ReLU.
    """
    if isinstance(other, (int, float)):
        other = torch.tensor(other, device=input.device, dtype=input.dtype)

    if out is None:
        if inplace:
            out = input
        else:
            out = torch.empty_like(input)

    n_elements = input.numel()
    grid = (triton.cdiv(n_elements, 1024),)
    block_size = 1024

    mul_relu_kernel[grid](
        input,
        other,
        out,
        n_elements,
        BLOCK_SIZE=block_size,
    )

    return out
