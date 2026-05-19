import torch
import triton
import triton.language as tl

# Kernel 1: argmax_kernel_1
@triton.jit
def argmax_kernel_1(
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
    amax_val, amax_idx = tl.max_and_argmax(inp_val, axis=0)
    mid_ptr = mid + pid * 2
    tl.store(mid_ptr, amax_val)
    tl.store(mid_ptr + 1, amax_idx)


# Kernel 2: argmax_kernel_2
@triton.jit
def argmax_kernel_2(mid, out, mid_size, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mid_ptrs = mid + offset * 2
    mask = offset < mid_size
    mid_val = tl.load(mid_ptrs, mask=mask, other=-float("inf"))
    mid_idx = tl.load(mid_ptrs + 1, mask=mask, other=-1)
    amax_val, amax_idx = tl.max_and_argmax(mid_val, axis=0)
    global_idx = tl.where(mid_val == amax_val, mid_idx, -1)
    global_idx = tl.max(global_idx)
    tl.store(out, amax_val)
    tl.store(out + 1, global_idx)


# Kernel 3: argmax_kernel
@triton.autotune(configs=cfggen(), key=["M", "N"])
@triton.jit
def argmax_kernel(
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

    _all_val = tl.full([BLOCK_M, BLOCK_N], value=-float("inf"), dtype=tl.float32)
    _all_idx = tl.full([BLOCK_M, BLOCK_N], value=-1, dtype=tl.int32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask & col_mask

        a_val = tl.load(inp + cols, mask, other=-float("inf")).to(tl.float32)
        a_idx = tl.load(inp + cols, mask, other=-1).to(tl.int32)
        _all_val = tl.where(a_val > _all_val, a_val, _all_val)
        _all_idx = tl.where(a_val > _all_val, a_idx, _all_idx)
    all_val = tl.max(_all_val, axis=1)[:, None]
    all_idx = tl.max(_all_idx, axis=1)[:, None]
    tl.store(out, all_val, row_mask)
    tl.store(out + 1, all_idx, row_mask)


# Function to call the kernels
def argmax(inp, dim=None, keepdim=False):
    if dim is None or len(dim) == 0:
        M = inp.numel()
        block_size = triton.next_power_of_2(math.ceil(math.sqrt(M)))
        mid_size = triton.cdiv(M, block_size)
        block_mid = triton.next_power_of_2(mid_size)
        dtype = inp.dtype
        mid = torch.empty((mid_size * 2,), dtype=dtype, device=inp.device)
        use_int64_index = not can_use_int32_index(inp)
        if not keepdim:
            out = torch.empty([2], dtype=dtype, device=inp.device)
        else:
            shape = list(inp.shape)
            for i in range(0, inp.dim()):
                shape[i] = 1
            out = torch.empty(shape + [2], dtype=dtype, device=inp.device)
        with torch.cuda.device(inp.device):
            argmax_kernel_1[(mid_size, 1)](
                inp, mid, M, block_size, INT64_INDEX=use_int64_index
            )
            argmax_kernel_2[(1, 1)](
                mid, out, mid_size, block_mid
            )
        return out[0] if not keepdim else out[0].squeeze(dim=dim)
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

        out = torch.empty(shape + [2], dtype=dtype, device=inp.device)

        grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
        with torch.cuda.device(inp.device):
            argmax_kernel[grid](inp, out, M, N, INT64_INDEX=use_int64_index)
        if not keepdim:
            out = out.squeeze(dim=dim)
        return out[:, 0], out[:, 1]


# Helper function to generate configurations for autotuning
def cfggen():
    block_m = [1, 2, 4, 8]
    configs = [
        triton.Config({"BLOCK_M": m, "BLOCK_N": 1024}, num_warps=4) for m in block_m
    ]
    return configs


# Helper function to determine if int32 index can be used
def can_use_int32_index(tensor):
    return tensor.numel() < 2**31


# Helper function to compress dimensions
def dim_compress(tensor, dims):
    shape = list(tensor.shape)
    new_shape = []
    compressed_size = 1
    for i, s in enumerate(shape):
        if i in dims:
            compressed_size *= s
        else:
            new_shape.append(s)
    new_shape.append(compressed_size)
    return tensor.view(new_shape)
