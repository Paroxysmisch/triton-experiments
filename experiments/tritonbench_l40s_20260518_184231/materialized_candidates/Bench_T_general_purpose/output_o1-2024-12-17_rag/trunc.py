import torch
import triton
import triton.language as tl

@triton.jit
def _trunc_kernel(
    input_ptr, output_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    values = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    # Truncate floats (positive -> floor, negative -> ceil)
    truncated_values = tl.where(values >= 0.0, tl.floor(values), tl.ceil(values))
    tl.store(output_ptr + offsets, truncated_values, mask=mask)


def trunc(input: torch.Tensor, *, out: torch.Tensor = None) -> torch.Tensor:
    """
    Returns a new tensor with the truncated integer values of the elements 
    of the input tensor. For integer inputs, returns a copy of the input 
    tensor (following array-api convention).
    
    Args:
        input (Tensor): the input tensor.
    
    Keyword args:
        out (Tensor, optional): the output tensor. If provided, the 
        result will be written to this tensor.
    
    Returns:
        Tensor: a new (or provided 'out') tensor with the truncated values.
    """
    # For integer inputs, return copy (array-api convention).
    if not input.is_floating_point():
        if out is None:
            return input.clone()
        else:
            return out.resize_as_(input).copy_(input)

    # For floating point inputs:
    if out is None:
        out = torch.empty_like(input)

    # Flatten and ensure contiguity for kernel launch
    in_flat = input.contiguous().view(-1)
    out_flat = out.contiguous().view(-1)

    n_elements = in_flat.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    _trunc_kernel[grid](
        input_ptr=in_flat,
        output_ptr=out_flat,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )

    # Reshape 'out' to original shape if needed
    if out is not None and out.dim() > 0:
        out.view_as_(input)

    return out
