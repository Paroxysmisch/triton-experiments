import torch
import triton
import triton.language as tl

@triton.jit
def max_kernel(
    input_ptr,
    output_ptr,
    indices_ptr,
    input_shape_ptr,
    output_shape_ptr,
    dim,
    keepdim,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    input_shape = tl.load(input_shape_ptr + tl.arange(0, len(input_ptr.shape)))
    output_shape = tl.load(output_shape_ptr + tl.arange(0, len(output_ptr.shape)))

    offset = 0
    strides = 1
    for i in range(len(input_ptr.shape)):
        if i != dim:
            offset += row_idx // strides * input_ptr.stride(i)
            strides *= input_ptr.shape[i]

    max_val = -float('inf')
    max_idx = -1

    for i in range(input_shape[dim]):
        val = tl.load(input_ptr + offset + i * input_ptr.stride(dim))
        if val > max_val:
            max_val = val
            max_idx = i
    
    output_offset = 0
    output_strides = 1
    for i in range(len(output_ptr.shape)):
        output_offset += (row_idx // output_strides) % output_shape[i] * output_ptr.stride(i)
        output_strides *= output_shape[i]
    
    tl.store(output_ptr + output_offset, max_val)
    tl.store(indices_ptr + output_offset, max_idx)


def max(input, dim, keepdim=False, *, out=None):
    input_shape = input.shape
    output_shape = list(input_shape)
    if keepdim:
        output_shape[dim] = 1
    else:
        output_shape.pop(dim)
    output_shape = tuple(output_shape)

    output = torch.empty(output_shape, dtype=input.dtype, device=input.device)
    indices = torch.empty(output_shape, dtype=torch.int64, device=input.device)

    grid = output.numel()
    block_size = min(grid, 1024)

    if out is not None:
        output, indices = out
        assert output.shape == output_shape, f"output tensor should have shape {output_shape}, but got {output.shape}"
        assert indices.shape == output_shape, f"indices tensor should have shape {output_shape}, but got {indices.shape}"
        assert indices.dtype == torch.int64, f"indices tensor should have dtype torch.int64, but got {indices.dtype}"


    max_kernel[(grid,)](
        input,
        output,
        indices,
        input.shape,
        output.shape,
        dim,
        keepdim,
        BLOCK_SIZE=block_size,
    )
    return output, indices
