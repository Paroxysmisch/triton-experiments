import triton
import triton.language as tl

# Kernel 1: Compute block-wise maximum values
@triton.jit
def max_kernel_1(X, mid, BLOCK_SIZE: tl.constexpr, N: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < N
    x = tl.load(X + offset, mask=mask, other=-float('inf'))
    max_val = tl.max(x, axis=0)
    tl.store(mid + pid, max_val)

# Kernel 2: Compute final maximum value from block-wise results
@triton.jit
def max_kernel_2(mid, result, N: tl.constexpr):
    x = tl.load(mid + tl.arange(0, N))
    max_val = tl.max(x, axis=0)
    tl.store(result, max_val)

# Kernel for max along a specific dimension
@triton.jit
def max_kernel(X, max_vals, max_indices, stride, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, M: tl.constexpr, N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offsets_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask_m = offsets_m < M
    mask_n = offsets_n < N
    x = tl.load(X + offsets_m[:, None] * stride + offsets_n[None, :], mask=mask_m[:, None] & mask_n[None, :], other=-float('inf'))
    max_val = tl.max(x, axis=1)
    max_idx = tl.argmax(x, axis=1)
    tl.store(max_vals + offsets_m, max_val, mask=mask_m)
    tl.store(max_indices + offsets_m, max_idx, mask=mask_m)

# Wrapper for computing max over entire tensor
def max(X, BLOCK_SIZE=1024):
    N = X.size
    mid = triton.empty([N // BLOCK_SIZE], dtype=X.dtype)
    result = triton.empty([1], dtype=X.dtype)
    grid = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    max_kernel_1[grid](X, mid, BLOCK_SIZE, N)
    max_kernel_2[1](mid, result, grid)
    return result[0]

# Wrapper for computing max along a specified dimension
def max_dim(X, dim, BLOCK_SIZE_M=128, BLOCK_SIZE_N=128):
    shape = X.shape
    M, N = shape[dim], shape[1 - dim]
    max_vals = triton.empty([M], dtype=X.dtype)
    max_indices = triton.empty([M], dtype=tl.int32)
    stride = shape[1] if dim == 0 else 1
    grid = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M, (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    max_kernel[grid](X, max_vals, max_indices, stride, BLOCK_SIZE_M, BLOCK_SIZE_N, M, N)
    return max_vals, max_indices
