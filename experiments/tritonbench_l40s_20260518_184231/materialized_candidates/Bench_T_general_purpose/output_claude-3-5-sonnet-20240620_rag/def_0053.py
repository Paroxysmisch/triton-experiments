import torch
import triton
import triton.language as tl
from torch import Tensor
import torch.nn.functional as F

@triton.jit
def mul_relu_kernel(input, other, output, n_elements):
    # Get the program ID (index of the current thread)
    idx = tl.program_id(0)
    
    # Load input and other tensors
    input_val = tl.load(input + idx)
    other_val = tl.load(other + idx)
    
    # Perform element-wise multiplication
    result = input_val * other_val
    
    # Apply ReLU function
    result = tl.maximum(result, 0.0)
    
    # Store the result in the output tensor
    tl.store(output + idx, result)

@torch.inference_mode()
def mul_relu(input: Tensor, other: Tensor, inplace: bool = False, out: Tensor = None) -> Tensor:
    """
    Performs element-wise multiplication of input and other, followed by ReLU.

    Args:
        input (Tensor): The input tensor to be multiplied.
        other (Tensor or Number): The tensor or number to multiply with `input`.
        inplace (bool, optional): If True, modifies `input` in-place, if possible. Default is False.
        out (Tensor, optional): The output tensor.

    Returns:
        Tensor: The output tensor after applying the multiplication and ReLU.
    """
    # Determine the number of elements
    n_elements = input.numel()
    
    # Create output tensor if not provided
    if out is None:
        out = torch.empty_like(input)

    # Launch the Triton kernel
    grid = (n_elements,)
    mul_relu_kernel[grid](input, other, out, n_elements)

    # If inplace is True, modify input tensor
    if inplace:
        input.copy_(out)
        return input

    return out
