import triton
import triton.language as tl

@triton.jit
def max_kernel_1(x_ptr, out_ptr, n, BLOCK_SIZE=64):
    pid = tl.program_id(axis=0)
    start = pid * BLOCK_SIZE
    end = start + BLOCK_SIZE
    mask = tl.mask(start<n, end>0)
    x = tl.load(x_ptr+start, mask=mask)
    max_val = tl.max(x)
    tl.store(out_ptr+start, max_val)

@triton.jit
def max_kernel_2(out_ptr, n, out_values):
    pid = tl.program_id(axis=0)
    start = pid * BLOCK_SIZE
    end = start + BLOCK_SIZE
    mask = tl.mask(start<n, end>0)
    out = tl.load(out_ptr+start, mask=mask)
    max_val = tl.max(out)
    tl.store(out_values+pid, max_val)

@triton.jit
def max_kernel(x_ptr, out_ptr, n, BLOCK_SIZE=64):
    pid_m = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)
    start_m = pid_m * BLOCK_SIZE
    start_k = pid_k * BLOCK_SIZE
    end_m = start_m + BLOCK_SIZE
    end_k = start_k + BLOCK_SIZE
    mask_m = tl.mask(start_m<n, end_m>0)
    mask_k = tl.mask(start_k<n, end_k>0)
    x = tl.load(x_ptr+start_m, mask=mask_m)
    x = tl.load(x, x_ptr+start_k, mask=mask_k)
    max_val = tl.max(x)
    tl.store(out_ptr+pid_m, max_val)

def max(x):
    n = x.numel()
    BLOCK_SIZE = 64
    grid = lambda meta: triton.grid(meta, BLOCK_SIZE)
    out = triton.empty((n,), dtype=x.dtype)
    mid = triton.empty((n,), dtype=x.dtype)
    max_kernel_1[grid](x.ptr, mid.ptr, n, BLOCK_SIZE)
    max_kernel_2[grid](mid.ptr, n, out.ptr)
    return out[0]

def max_dim(x, dim):
    assert 0 <= dim < x.ndim
    shape = list(x.shape)
    M = functools.reduce(operator.mul, shape[:dim], 1)
    K = functools.reduce(operator.mul, shape[dim+1:], 1)
    shape[dim] = 1
    y = x.reshape(M, -1)
    out = triton.empty((M,), dtype=x.dtype)
    max_kernel[grid](y.ptr, out.ptr, y.numel())
    return out.reshape(shape[:dim]+shape[dim+1:])
