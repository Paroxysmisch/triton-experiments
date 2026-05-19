import torch
import triton
import triton.language as tl
import math

# Kernel 1: amax_kernel_1
@triton.jit
def amax_kernel_1(
    inp,
    mid,
    M,
    BLOCK_SIZE: tl.constexpr,
    INT64_INDEX: tl.constexpr = False,
):
    pid = tl.program_id(0)
    if INT64_INDEX:
        pid = pid.to(tl.int64)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inp_ptrs = inp + offset
    mask = offset < M
    inp_val = tl.load(inp_ptrs, mask=mask, other=-float("inf"))
    amax_val = tl.max(inp_val)
    mid_ptr = mid + pid
    tl.store(mid_ptr, amax_val)

# Kernel 2: amax_kernel_2
@triton.jit
def amax_kernel_2(mid, out, mid_size, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mid_ptrs = mid + offset
    mask = offset < mid_size
    mid_val = tl.load(mid_ptrs, mask=mask, other=-float("inf"))
    amax_val = tl.max(mid_val)
    tl.store(out, amax_val)

# Kernel 3: amax_kernel
@triton.autotune(configs=cfggen(), key=["M", "N"])
@triton.jit
def amax_kernel(
    inp,
    out,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    INT64_INDEX: tl.constexpr = False,
):
    pid = tl.program_id(0)
    if INT64_INDEX:
        pid = pid.to(tl.int64)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    inp = inp + rows * N
    out = out + rows
    row_mask = rows < M

    _all = tl.full([BLOCK_M, BLOCK_N], value=-float("inf"), dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask and col_mask

        a = tl.load(inp + cols, mask, other=-float("inf")).to(tl.float32)
        _all = tl.maximum(_all, a)
    all = tl.max(_all, axis=1)[:, None]
    tl.store(out, all, row_mask)

# Function to call the kernels
def amax(inp, dim=None, keepdim=False):
    if dim is None or len(dim) == 0:
        M = inp.numel()
        block_size = triton.next_power_of_2(math.ceil(math.sqrt(M)))
        mid_size = triton.cdiv(M, block_size)
        block_mid = triton.next_power_of_2(mid_size)
        dtype = inp.dtype
        mid = torch.empty((mid_size,), dtype=dtype, device=inp.device)
        use_int64_index = not can_use_int32_index(inp)
        if not keepdim:
            out = torch.empty([], dtype=dtype, device=inp.device)
        else:
            shape = list(inp.shape)
            for i in range(0, inp.dim()):
                shape[i] = 1
            out = torch.empty(shape, dtype=dtype, device=inp.device)
        with torch.cuda.device(inp.device):
            amax_kernel_1[(mid_size, 1)](
                inp, mid, M, block_size, INT64_INDEX=use_int64_index
            )
            amax_kernel_2[(1, 1)](
                mid, out, mid_size, block_mid
            )  # max block size is 128k, so mid does not require int64 index
        return out
    else:
        if isinstance(dim, int):
            dim = [dim]
        assert all((i >= -inp.ndim and i < inp.ndim) for i in dim), "Invalid dim"
        dtype = inp.dtype

        shape = list(inp.shape)
        dim = [d % inp.ndim for d in dim]
        inp = dim_compress(inp, dim)
        use_int64_index = not can_use_int32_index(inp)
        N = 1
        for i in dim:
            N *= shape[i]
            shape[i] = 1
        M = inp.numel() // N

        out = torch.empty(shape, dtype=dtype, device=inp.device)

        grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
        with torch.cuda.device(inp.device):
            amax_kernel[grid](inp, out, M, N, INT64_INDEX=use_int64_index)
        if not keepdim:
            out = out.squeeze(dim=dim)
        return out

# Helper function to generate configurations for autotuning
def cfggen():
    block_m = [1, 2, 4, 8]
    configs = [
        triton.Config({"BLOCK_M": m, "BLOCK_N": 1024}, num_warps=4) for m in block_m
    ]
    return configs

# Helper function to compress dimensions
def dim_compress(tensor, dims):
    shape = list(tensor.shape)
    for d in sorted(dims, reverse=True):
        shape.pop(d)
    return tensor.view(*shape)

# Helper function to check if int32 index can be used
def can_use_int32_index(tensor):
    return tensor.numel() < 2**31

# Function to find max values along a specified dimension
def amax_dim(inp, dim, keepdim=False):
    return amax(inp, dim=dim, keepdim=keepdim)
