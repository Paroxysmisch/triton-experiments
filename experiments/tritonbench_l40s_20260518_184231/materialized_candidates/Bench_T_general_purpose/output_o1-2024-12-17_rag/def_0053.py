import torch
import triton
import triton.language as tl
from torch import Tensor
from triton.runtime.jit import get_cuda_stream


@triton.jit
def _mul_relu_kernel(
    input_ptr,
    other_ptr,
    output_ptr,
    n_elements,
    scalar_val,
    is_scalar: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    start_offset = pid * BLOCK_SIZE
    offsets = start_offset + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)

    # If 'other' is a scalar, use scalar_val. Otherwise, load from other_ptr.
    if is_scalar:
        m = x * scalar_val
    else:
        y = tl.load(other_ptr + offsets, mask=mask, other=0.0)
        m = x * y

    # ReLU
    m = tl.maximum(m, 0)

    tl.store(output_ptr + offsets, m, mask=mask)


def mul_relu(input, other, inplace=False, out=None) -> Tensor:
    """
    Performs element-wise multiplication of 'input' and 'other', then applies ReLU.

    Args:
        input (Tensor): The input tensor.
        other (Tensor or Number): The tensor or number to multiply with 'input'.
        inplace (bool, optional): If True, modifies 'input' in-place if possible. Default is False.
        out (Tensor, optional): Output tensor. If provided, result is placed in 'out'.
    """

    if not isinstance(input, Tensor):
        raise TypeError("'input' must be a torch Tensor")

    # Determine the output destination
    if out is not None:
        if not isinstance(out, Tensor):
            raise TypeError("'out' must be a torch Tensor or None")
        if out.shape != input.shape:
            raise ValueError("'out' must have the same shape as 'input'")
        output = out
    else:
        output = input if inplace else torch.empty_like(input)

    n_elements = input.numel()

    # Flatten tensors for kernel launch
    flat_input = input.view(-1)
    flat_output = output.view(-1)
    flat_other = None

    is_tensor_other = isinstance(other, Tensor)
    if is_tensor_other:
        flat_other = other.view(-1)
        # Broadcast check (only to single value or exact shape here)
        if flat_other.numel() not in [1, n_elements]:
            raise ValueError(
                "'other' must be a number or a tensor broadcastable to 'input'"
            )
        is_scalar = (flat_other.numel() == 1)
        scalar_val = flat_other[0].item() if is_scalar else 0.0
    else:
        # 'other' is a scalar
        is_scalar = True
        scalar_val = float(other)  # cast to float

    # Launch kernel
    device = flat_input.device
    stream = get_cuda_stream(device.index) if device.type == 'cuda' else None

    BLOCK_SIZE = 1024
    grid = lambda meta: ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    # Pass a valid pointer for other_ptr only if 'other' is tensor and not scalar
    _mul_relu_kernel[grid](
        flat_input,
        flat_other if (is_tensor_other and not is_scalar) else flat_input,
        flat_output,
        n_elements,
        scalar_val,
        is_scalar,
        BLOCK_SIZE,
        num_warps=4,
        num_stages=2,
        stream=stream,
    )

    return output
