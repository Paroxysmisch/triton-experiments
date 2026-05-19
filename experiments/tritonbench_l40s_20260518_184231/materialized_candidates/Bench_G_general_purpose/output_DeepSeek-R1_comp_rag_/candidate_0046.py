import torch
import triton
import triton.language as tl
import math

# Helper function to check if 32-bit indexing is sufficient
def can_use_int32_index(tensor):
    return tensor.numel() <= (1 << 31)

# Autotune configurations for max_kernel
def cfggen():
    block_m = [1, 2, 4, 8, 16, 32]
    configs = [
        triton.Config({'BLOCK_M': m, 'BLOCK_N': 1024}, num_warps=4)
        for m in block_m
    ]
    return configs

# Kernel 1: Compute intermediate max values
@triton.jit
def max_kernel_1(
    input_ptr,
    mid_ptr,
    num_elements,
    BLOCK_SIZE: tl.constexpr,
    INT64_INDEX: tl.constexpr = False,
):
    pid = tl.program_id(0)
    if INT64_INDEX:
        pid = pid.to(tl.int64)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    values = tl.load(input_ptr + offsets, mask=mask, other=-float('inf'))
    max_val = tl.max(values)
    tl.store(mid_ptr + pid, max_val)

# Kernel 2: Compute final max from intermediates
@triton.jit
def max_kernel_2(
    mid_ptr,
    output_ptr,
    mid_size,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    mid_ptrs = mid_ptr + offsets
    mask = offsets < mid_size
    values = tl.load(mid_ptrs, mask=mask, other=-float('inf'))
    max_val = tl.max(values)
    tl.store(output_ptr, max_val)

# Kernel 3: Multi-dimensional max along a specified dimension
@triton.autotune(configs=cfggen(), key=['M', 'N'])
@triton.jit
def max_kernel(
    input_ptr,
    output_ptr,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    INT64_INDEX: tl.constexpr = False,
):
    pid = tl.program_id(0)
    if INT64_INDEX:
        pid = pid.to(tl.int64)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < M
    acc = tl.full((BLOCK_M,), -float('inf'), dtype=tl.float32)
    input_ptr += rows[:, None] * N
    for block_n in range(0, N, BLOCK_N):
        cols = block_n + tl.arange(0, BLOCK_N)
        col_mask = cols < N
        mask = row_mask[:, None] & col_mask[None, :]
        vals = tl.load(input_ptr + cols, mask=mask, other=-float('inf'), eviction_policy='evict_first')
        acc = tl.maximum(acc, tl.max(vals, axis=1))
    tl.store(output_ptr + rows, acc, mask=row_mask)

# Wrapper for overall max
def max(input, keepdim=False):
    if input.dim() == 0:
        return input
    M = input.numel()
    block_size = triton.next_power_of_2(int(math.sqrt(M)))
    mid_size = (M + block_size - 1) // block_size
    dtype = input.dtype
    device = input.device
    mid = torch.empty(mid_size, dtype=dtype, device=device)
    use_int64_index = not can_use_int32_index(input)
    grid1 = (mid_size,)
    max_kernel_1[grid1](input, mid, M, block_size, INT64_INDEX=use_int64_index)
    block_mid = triton.next_power_of_2(mid_size)
    out_shape = [1]*input.dim() if keepdim else []
    out = torch.empty(out_shape, dtype=dtype, device=device)
    max_kernel_2[(1,)](mid, out, mid_size, block_mid)
    return out

# Wrapper for max along a dimension
def max_dim(input, dim, keepdim=False):
    if dim is None:
        return max(input, keepdim=keepdim)
    dim = dim if dim >= 0 else input.dim() + dim
    assert 0 <= dim < input.dim(), "dim out of range"
    shape = list(input.shape)
    M = 1
    for s in shape[:dim]:
        M *= s
    K = shape[dim]
    N = 1
    for s in shape[dim+1:]:
        N *= s
    input_flat = input.view(M, K, N).permute(0, 2, 1).contiguous().view(M*N, K)
    out_flat = torch.empty((M*N, 1), dtype=input.dtype, device=input.device)
    use_int64_index = not can_use_int32_index(input_flat)
    grid = lambda meta: (triton.cdiv(M*N, meta['BLOCK_M']),)
    max_kernel[grid](input_flat, out_flat, M*N, K, INT64_INDEX=use_int64_index)
    out_shape = list(shape)
    if keepdim:
        out_shape[dim] = 1
    else:
        out_shape.pop(dim)
    return out_flat.view(out_shape)
