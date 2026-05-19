import torch
import triton
import triton.language as tl
from triton.runtime import driver

@triton.jit
def _fused_tile_exp_kernel(input, output, stride_input, stride_output, n_dims_interleave, n_tiles, BLOCK_SIZE: tl.constexpr,
                           num_stages: tl.constexpr):
    pid = tl.program_id(axis=0)

    # This is equivalent to:
    # for i in range(0, n_tiles):
    #   offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    #   mask = offsets < n_tiles
    #   input_ptrs = input + offsets
    #   outputs_ptrs = output + offsets
    #   tiled_input = tl.load(input_ptrs, mask=mask, other=0)
    #   tiled_output = tl.exp(tiled_input)
    #   tl.store(outputs_ptrs, tiled_output, mask=mask)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tiles_mask = offsets < n_tiles
    input_ptrs = offsets
    expanded_input_ptrs = input_ptrs + tl.expand_dims(input_ptr_repeated_dims_offsets, axis=0)[:, :]
    expanded_output_ptrs = output_ptrs + tl.expand_dims(output_ptr_repeated_dims_offsets, axis=1)[:, :]
    tiled_input = tl.load(expanded_input_ptrs, mask=tiles_mask, other=0)
    tiled_output = tl.exp(tiled_input)
    tl.store(expanded_output_ptrs, tiled_output, mask=tiles_mask)

def fused_tile_exp(input, dims, *, out=None):
    n_dims = input.ndim
    n_dims_to_repeat = len(dims)
    n_tiles = 1

    for dim_value in dims:
        n_tiles *= dim_value

    tiles_shape = list()

    for idx in range(n_dims_to_repeat):
        tiles_shape.append(dims[idx])
        tiles_shape.append(input.shape[idx])

    for idx in range(n_dims - n_dims_to_repeat):
        tiles_shape.append(1)
        tiles_shape.append(input.shape[n_dims_to_repeat + idx])

    tiles_strides = [input.strides[-1]] + list(input.stride()) * 2

    for idx in range(n_dims_to_repeat):
        tiles_strides[2 * idx + 1] *= dims[idx]

    for idx in range(n_dims - n_dims_to_repeat):
        tiles_strides[2 * idx + 2] *= dims[idx]

    if out is None:
        out = torch.empty(tiles_shape, dtype=input.dtype, device=input.device)
    else:
        assert out.shape == tiles_shape
        assert out.dtype == input.dtype
        assert out.device.type == input.device.type

    n_dims_interleave = 1

    while n_dims_interleave <= n_dims_to_repeat and n_dims_interleave * 2 <= triton.next_power_of_2(n_tiles):
        n_dims_interleave *= 2

    if n_dims_interleave < n_dims_to_repeat or n_dims_interleave * 2 > n_tiles:
        _fused_tile_exp_kernel[(n_tiles, )](input, out, input.stride(0), out.stride(0), n_dims_interleave, n_tiles,
                                            BLOCK_SIZE=n_dims_to_repeat,
                                            num_stages=1)
    else:
        _fused_tile_exp_kernel[(n_tiles, ), num_warps=1](input, out, input.stride(0), out.stride(0),
                                                         n_dims_interleave, n_tiles)

    return out
