import torch
import triton
import triton.language as tl
from typing import Optional, Tuple


@triton.jit
def max_1d_kernel(in_ptr, out_value_ptr, out_index_ptr, n_elements,
                  BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    # load data to SRAM
    x = tl.load(in_ptr + offsets, mask=mask)

    # compute max
    value, index = tl.max(x, axis=0, return_indices=True)

    # write-back
    out_offsets = pid
    out_mask = pid < 1
    tl.store(out_value_ptr + out_offsets, value, mask=out_mask)
    tl.store(out_index_ptr + out_offsets, index, mask=out_mask)


def max(input: torch.Tensor, dim: int = None, keepdim: bool = False,
        out: Optional[Tuple[Optional[torch.Tensor],
                            Optional[torch.Tensor]]] = None) -> Tuple[
                                torch.Tensor, torch.Tensor]:
    assert dim >= -input.ndim and dim < input.ndim, "Invalid dim"
    dim = dim % input.ndim
    shape = list(input.shape)
    n_elements = shape[dim]
    shape[dim] = 1
    output_shape = shape if keepdim else [i for i in shape if i != 1]

    if out is None:
        out_value = torch.empty(output_shape, dtype=input.dtype, device=input.device)
        out_index = torch.empty(output_shape, dtype=torch.int64, device=input.device)
    elif out[0] is None:
        out_value = torch.empty(output_shape, dtype=input.dtype, device=input.device)
        out_index = torch.empty(output_shape, dtype=torch.int64, device=input.device)
    elif out[1] is None:
        out_value = torch.empty(output_shape, dtype=input.dtype, device=input.device)
        out_index = torch.empty(output_shape, dtype=torch.int64, device=input.device)
    else:
        out_value, out_index = out

    assert out_value.shape == output_shape, "Invalid out[0]"
    assert out_index.shape == output_shape, "Invalid out[1]"

    BLOCK_SIZE = triton.next_power_of_2(n_elements)
    grid = (1, )

    max_1d_kernel[grid](input, out_value, out_index, n_elements,
                        BLOCK_SIZE=BLOCK_SIZE)

    if not keepdim:
        out_value.squeeze_(dim)
        out_index.squeeze_(dim)

    return out_value, out_index
