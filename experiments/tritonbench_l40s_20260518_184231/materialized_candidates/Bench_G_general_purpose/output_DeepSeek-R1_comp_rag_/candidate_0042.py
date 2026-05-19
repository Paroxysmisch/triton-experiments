import triton
import triton.language as tl
import torch
from typing import Optional, Union

@triton.jit
def pow_func_scalar_tensor_kernel_rank_1(
    input_ptr,
    output_ptr,
    exponent,
    n_elements,
    input_stride,
    output_stride,
    TILE_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * TILE_SIZE
    offsets = block_start + tl.arange(0, TILE_SIZE)
    mask = offsets < n_elements

    input_block_ptr = tl.make_block_ptr(
        base=input_ptr,
        shape=(n_elements,),
        strides=(input_stride,),
        offsets=(block_start,),
        block_shape=(TILE_SIZE,),
        order=(0,)
    )
    input_vals = tl.load(input_block_ptr, mask=mask, other=0.0)

    output_vals = tl.pow(input_vals, exponent)

    output_block_ptr = tl.make_block_ptr(
        base=output_ptr,
        shape=(n_elements,),
        strides=(output_stride,),
        offsets=(block_start,),
        block_shape=(TILE_SIZE,),
        order=(0,)
    )
    tl.store(output_block_ptr, output_vals, mask=mask)

def pow_func_scalar_tensor_wrapper_rank_1(
    input: Union[torch.Tensor, 'StridedBuffer'],
    exponent: float,
    output: Optional[Union[torch.Tensor, 'StridedBuffer']] = None,
) -> Union[torch.Tensor, 'StridedBuffer']:
    assert input.ndim == 1, "Input tensor must be 1-dimensional"
    if output is None:
        output = torch.empty_like(input) if isinstance(input, torch.Tensor) else input.__class__.empty_like(input)
    else:
        assert output.shape == input.shape, "Output shape must match input shape"

    n_elements = input.shape[0]
    device = input.device if isinstance(input, torch.Tensor) else input.buffer.device

    def get_ptr_and_stride(x):
        if isinstance(x, torch.Tensor):
            return x.data_ptr(), x.stride(0)
        else:  # Assume StridedBuffer-like
            return x.buffer.data_ptr(), x.strides[0]

    input_ptr, input_stride = get_ptr_and_stride(input)
    output_ptr, output_stride = get_ptr_and_stride(output)

    max_tile_size = 1024
    tile_size = min(max_tile_size, triton.next_power_of_2(n_elements))
    tile_size = max(tile_size, 16)  # Ensure a minimum tile size

    grid = (triton.cdiv(n_elements, tile_size),)
    num_warps = tile_size // 32 if tile_size >= 32 else 1

    pow_func_scalar_tensor_kernel_rank_1[grid](
        input_ptr,
        output_ptr,
        exponent,
        n_elements,
        input_stride,
        output_stride,
        TILE_SIZE=tile_size,
        BLOCK_SIZE=tile_size,
        num_warps=num_warps,
    )

    return output
