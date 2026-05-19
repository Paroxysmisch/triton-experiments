import torch
import triton
import triton.language as tl

@triton.jit
def ifftshift_single_dim_kernel(
    output_ptr,
    input_ptr,
    input_numel,
    dim_size,
    dim_stride,
    shift,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < input_numel

    for idx in tl.range(BLOCK_SIZE):
        if mask[idx]:
            linear_idx = offsets[idx]
            index_along_dim = (linear_idx // dim_stride) % dim_size
            adjusted_index = (index_along_dim + shift) % dim_size
            delta = (adjusted_index - index_along_dim) * dim_stride
            original_linear_idx = linear_idx + delta
            val = tl.load(input_ptr + original_linear_idx)
            tl.store(output_ptr + linear_idx, val)

def _ifftshift_single_dim(input, dim):
    dim = dim if dim >= 0 else input.ndim + dim
    assert 0 <= dim < input.ndim, "dim out of range"
    
    dim_size = input.shape[dim]
    if dim_size <= 1:
        return input  # No shift needed
    
    shift = (dim_size + 1) // 2
    if shift == 0:
        return input  # No shift needed
    
    output = torch.empty_like(input)
    dim_stride = input.stride()[dim]
    numel = input.numel()
    
    BLOCK_SIZE = 128
    grid = (triton.cdiv(numel, BLOCK_SIZE),)
    
    ifftshift_single_dim_kernel[grid](
        output,
        input,
        numel,
        dim_size,
        dim_stride,
        shift,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output

def ifftshift(input, dim=None):
    if dim is None:
        dim = tuple(range(input.ndim))
    elif isinstance(dim, int):
        dim = (dim,)
    else:
        dim = tuple(dim)
    
    output = input
    for d in dim:
        output = _ifftshift_single_dim(output, d)
    return output
