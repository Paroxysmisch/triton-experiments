import torch
import triton
import triton.language as tl

@triton.jit
def max_kernel_1(inp, mid, M, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inp_ptrs = inp + offset
    mask = offset < M
    inp_val = tl.load(inp_ptrs, mask=mask, other=-float('inf'))
    max_val = tl.max(inp_val)
    mid_ptr = mid + pid
    tl.store(mid_ptr, max_val)

@triton.jit
def max_kernel_2(mid, out, mid_size):
    mid_ptrs = mid + tl.arange(0, mid_size)
    mid_val = tl.load(mid_ptrs, mask=None, other=-float('inf'))
    max_val = tl.max(mid_val)
    tl.store(out, max_val)

@triton.jit
def max_kernel(inp, out, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    offset_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offset_k = pid_k * BLOCK_N + tl.arange(0, BLOCK_N)
    inp_ptrs = inp + offset_m[:, None] * N + offset_k
    mask = (offset_m < M) & (offset_k < N)
    inp_val = tl.load(inp_ptrs, mask=mask, other=-float('inf')).to(tl.float32)
    max_val, _ = tl.max(inp_val, axis=1)
    out_ptrs = out + offset_m[:, None]
    tl.store(out_ptrs, max_val, mask=(offset_m < M))

def max(inp):
    M = inp.numel()
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(M)))
    mid_size = triton.cdiv(M, block_size)
    dtype = inp.dtype
    mid = torch.empty((mid_size,), dtype=dtype, device=inp.device)
    out = torch.empty([], dtype=dtype, device=inp.device)
    max_kernel_1[(mid_size, 1)](inp, mid, M, block_size)
    max_kernel_2[(1, 1)](mid, out, mid_size)
    return out

def max_dim(inp, dim):
    if isinstance(dim, int):
        dim = [dim]
    assert ((i >= -inp.ndim and i < inp.ndim) for i in dim), "Invalid dim"
    dtype = inp.dtype
    shape = list(inp.shape)
    dim = [d % inp.ndim for d in dim]
    inp = dim_compress(inp, dim)
    M = 1
    for i in dim:
        M *= shape[i]
        shape[i] = 1
    N = 1
    for i in range(inp.ndim):
        if i not in dim:
            N *= inp.shape[i]
    out = torch.empty(shape, dtype=dtype, device=inp.device)
    block_m = triton.next_power_of_2(M)
    block_n = triton.next_power_of_2(N // block_m)
    max_kernel[M//block_m, N//block_n](inp, out, M, N, BLOCK_M=block_m, BLOCK_N=block_n)
    if len(dim) > 0:
        out = out.reshape(shape)
    return out
