import torch
import triton
import triton.language as tl

# Triton kernel for truncating elements of a tensor
@triton.jit
def trunc_kernel(
    input_ptr: tl.tensor, output_ptr: tl.tensor,
    size: tl.constexpr, element_size: tl.constexpr
):
    """Truncate the elements of the input tensor and store the result in the output tensor.

    Args:
        input_ptr: Pointer to the input tensor.
        output_ptr: Pointer to the output tensor.
        size: Number of elements in the input tensor.
        element_size: Size of each element in bytes.
    """
    pid = tl.program_id(axis=0)
    block_size = 128  # Number of elements processed by each block
    block_start = pid * block_size
    offsets = block_start + tl.arange(0, block_size)
    mask = offsets < size

    input_block = tl.load(input_ptr + offsets * element_size, mask=mask)
    output_block = tl.where(input_block >= 0, tl.floor(input_block), tl.ceil(input_block))
    tl.store(output_ptr + offsets * element_size, output_block, mask=mask)

# Wrapper function for the trunc kernel
def trunc(input: torch.Tensor, *, out: torch.Tensor = None) -> torch.Tensor:
    """Returns a new tensor with the truncated integer values of the elements of the input tensor.
    For integer inputs, it follows the array-api convention of returning a copy of the input tensor.

    Args:
        input (Tensor): the input tensor.
        out (Tensor, optional): the output tensor.

    Returns:
        Tensor: the truncated tensor.
    """
    if input.dtype in [torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8]:
        if out is not None:
            out.copy_(input)
        else:
            out = input.clone()
        return out

    if out is None:
        out = torch.empty_like(input, device=input.device, dtype=input.dtype)

    grid = (triton.cdiv(input.numel(), 128),)
    trunc_kernel[grid](
        input_ptr=input, output_ptr=out,
        size=input.numel(), element_size=input.element_size()
    )

    return out
