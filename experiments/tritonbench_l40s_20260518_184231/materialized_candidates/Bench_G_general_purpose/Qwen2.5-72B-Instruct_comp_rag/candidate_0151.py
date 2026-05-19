import torch
import triton
import triton.language as tl

# Kernel 1: argmax_kernel_1
@triton.jit
def argmax_kernel_1(
    inp,
    mid_value,
    mid_index,
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
    amax_idx = tl.argmax(inp_val)
    mid_value_ptr = mid_value + pid
    mid_index_ptr = mid_index + pid
    tl.store(mid_value_ptr, amax_val)
    tl.store(mid_index_ptr, amax_idx)

# Kernel 2: argmax_kernel_2
@triton.jit
def argmax_kernel_2(
    mid_value,
    mid_index,
    out,
    mid_size,
    BLOCK_MID: tl.constexpr,
):
    offset = tl.arange(0, BLOCK_MID)
    mid_value_ptrs = mid_value + offset
    mid_index_ptrs = mid_index + offset
    mask = offset < mid_size
    mid_val = tl.load(mid_value_ptrs, mask=mask, other=-float("inf"))
    mid_idx = tl.load(mid_index_ptrs, mask=mask, other=-1)
    amax_val = tl.max(mid_val)
    amax_idx = tl.argmax(mid_val)
    out_val_ptr = out + 0
    out_idx_ptr = out + 1
    tl.store(out_val_ptr, amax_val)
    tl.store(out_idx_ptr, mid_idx[amax_idx])

# Kernel 3: argmax_kernel
@triton.autotune(configs=cfggen(), key=["M", "N"])
@triton.jit
def argmax_kernel(
    inp,
    out,
    M,
    N,
    K,
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

    _all_val = tl.full([BLOCK_M, BLOCK_N], value=-float("inf"), dtype=tl.float32)
    _all_idx = tl.full([BLOCK_M, BLOCK_N], value=-1, dtype=tl.int32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask and col_mask

        a_val = tl.load(inp + cols, mask, other=-float("inf")).to(tl.float32)
        a_idx = tl.load(inp + cols, mask, other=-1).to(tl.int32)
        _all_val = tl.maximum(_all_val, a_val)
        _all_idx = tl.where(_all_val == a_val, a_idx, _all_idx)
    all_val = tl.max(_all_val, axis=1)[:, None]
    all_idx = tl.argmax(_all_val, axis=1)[:, None]
    tl.store(out, all_idx, row_mask)

# Function to call the kernels
def argmax(inp, dim=None, keepdim=False):
    if dim is None or len(dim) == 0:
        M = inp.numel()
        block_size = triton.next_power_of_2(math.ceil(math.sqrt(M)))
        mid_size = triton.cdiv(M, block_size)
        block_mid = triton.next_power_of_2(mid_size)
        dtype = inp.dtype
        mid_value = torch.empty((mid_size,), dtype=dtype, device=inp.device)
        mid_index = torch.empty((mid_size,), dtype=torch.int64, device=inp.device)
        use_int64_index = not can_use_int32_index(inp)
        if not keepdim:
            out = torch.empty([2], dtype=torch.int64, device=inp.device)
        else:
            shape = list(inp.shape)
            for i in range(0, inp.dim()):
                shape[i] = 1
            out = torch.empty(shape + [2], dtype=torch.int64, device=inp.device)
        with torch.cuda.device(inp.device):
            argmax_kernel_1[(mid_size, 1)](
                inp, mid_value, mid_index, M, block_size, INT64_INDEX=use_int64_index
            )
            argmax_kernel_2[(1, 1)](
                mid_value, mid_index, out, mid_size, block_mid
            )
        return out[1] if not keepdim else out[..., 1]
    else:
        if isinstance(dim, int):
            dim = [dim]
        assert ((i >= -inp.ndim and i < inp.ndim) for i in dim), "Invalid dim"
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
        K = inp.numel() // (M * N)

        out = torch.empty(shape, dtype=torch.int64, device=inp.device)

        grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
        with torch.cuda.device(inp.device):
            argmax_kernel[grid](inp, out, M, N, K, INT64_INDEX=use_int64_index)
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

# Helper function to check if int32 index can be used
def can_use_int32_index(tensor):
    return tensor.numel() <= 2**31 - 1

# Helper function to compress dimensions
def dim_compress(tensor, dim):
    shape = list(tensor.shape)
    for d in dim:
        shape[d] = 1
    return tensor.view(shape)
