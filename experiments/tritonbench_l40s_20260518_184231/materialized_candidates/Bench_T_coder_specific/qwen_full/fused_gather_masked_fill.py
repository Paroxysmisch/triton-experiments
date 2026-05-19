import torch
import triton
import triton.language as tl
from torch.distributed._tensor.experimental import local_map

@triton.jit
def _fused_gather_masked_fill_kernel(
    input, index, mask, value, dim, out, I, M, N, input_strides, index_strides, mask_strides, out_strides, IS_SCALAR
):
    pid = tl.program_id(0)
    # Compute offsets for the batch dimensions
    batch_offsets = pid * N
    # Compute offsets for the index dimension
    idx_stride = tl.load(index_strides + dim)
    idx_offsets = tl.arange(0, M) * idx_stride
    index_offsets = batch_offsets + idx_offsets
    # Load index values
    idx = tl.load(index + index_offsets)
    cond = tl.logical_and(idx < N, idx >= 0)
    idx = tl.where(cond, idx, 0)
    # Compute offsets for the remaining dimensions
    n = N
    for i in range(1, I):
        dim = dim % (tl.num_programs(0) // N)
        pid = tl.program_id(i)
        stride = tl.load(input_strides + dim)
        dim_offsets = pid * stride
        dim_sizes = tl.load(input_strides + dim + 1)
        dim_size = tl.minimum(n, dim_sizes)
        n = n // dim_size
        dim_idx = tl.arange(0, dim_size)
        offsets = dim_idx + dim_offsets
        mask = offsets < dim_sizes * dim_size
        cur_idx = tl.load(index + index_offsets + offsets, mask=mask)
        idx = idx * dim_size + cur_idx
    idx = idx + batch_offsets
    # Load input and apply mask
    cur_input = tl.load(input + idx, mask=cond)
    cur_mask = tl.load(mask + batch_offsets + tl.arange(0, M), mask=cond)
    cur_value = tl.full([M], value, tl.int64)
    if IS_SCALAR:
        cur_value = cur_value + 0 * cur_input
    cur_out = tl.where(cur_mask, cur_value, cur_input)
    # Store result
    tl.store(out + idx, cur_out, mask=cond)

def fused_gather_masked_fill(
    input, dim, index, mask, value, *, sparse_grad=False, out=None
):
    I, M, N = input.ndim, index.shape[-1], input.shape[dim]
    if out is None:
        out = torch.empty_like(input)
    else:
        assert out.shape == input.shape
    input_strides = list(input.stride())
    index_strides = list(index.stride())
    mask_strides = list(mask.stride())
    out_strides = list(out.stride())
    # Broadcast mask to output shape
    for i in range(I - len(mask_strides)):
        mask_strides.append(1)
    mask = mask.reshape(mask_strides)
    # Determine if value is a scalar
    IS_SCALAR = isinstance(value, (float, int))
    if IS_SCALAR:
        value = torch.tensor(value, device=input.device, dtype=torch.int64)
    # Launch Triton kernel
    grid = (
        triton.cdiv(input.numel() // N, N),
        *input.shape[:-1],
    )
    kwargs = [
        input,
        index,
        mask,
        value,
        dim,
        out,
        I,
        M,
        N,
        input_strides,
        index_strides,
        mask_strides,
        out_strides,
        IS_SCALAR,
    ]
    _fused_gather_masked_fill_kernel[grid](*kwargs)
    if sparse_grad:
        out = local_map(out, requires_grad=True)
    return out
