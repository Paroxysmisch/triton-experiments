import torch
import triton
import triton.language as tl

def can_use_int32_index(n: int) -> bool:
    return n <= torch.iinfo(torch.int32).max

def argmax(
    inp,
    dim=None,
    keepdim=False,
    *,
    dtype=None,
) -> torch.Tensor:
    if dim is None:
        if inp.numel() == 0:
            return torch.tensor([], dtype=torch.int64, device=inp.device)

        if inp.ndim == 1:
            inp = inp.view(-1, 1)

        inp_flatten = inp.reshape(-1)
        sem = triton.cdiv(inp_flatten.numel(), 2)
        mid_shape = inp.shape[:-1]
        mid_dtype = torch.int64 if can_use_int32_index(inp_flatten.numel()) else torch.int64
        mid_value = torch.empty(mid_shape, dtype=inp.dtype, device=inp.device)
        mid_index = torch.empty(mid_shape, dtype=mid_dtype, device=inp.device)
        argmax_kernel_1[(sem,)](inp_flatten, mid_value, mid_index, inp_flatten.numel(), BLOCK_SIZE=sem, INT64_INDEX=not can_use_int32_index(inp_flatten.numel()))
        argmax_kernel_2[(1,)](mid_value, mid_index, mid_value, mid_index.numel(), BLOCK_MID=mid_value.numel())
        ret = mid_value.reshape([-1])
        if not keepdim:
            ret = torch.reshape(ret, list(mid_shape))
        return ret

    if dim < 0:
        dim = dim + inp.ndim

    assert inp.shape[dim] != 0, "encounter zero-sized dimension"

    inp = inp.transpose(0, dim).contiguous()
    N = 1
    for i in range(1, inp.dim() - 1):
        N *= inp.size(i)
    M = inp.size(-1)

    K = triton.cdiv(M, BLOCK_N)
    sem = triton.cdiv(N, BLOCK_M)

    inp = inp.view(-1, M)

    if dtype is None:
        dtype = inp.dtype
    elif dtype != inp.dtype:
        inp = inp.to(dtype)

    out = torch.empty(N, dtype=inp.dtype, device=inp.device)

    grid = (sem,)
    argmax_kernel[grid](inp, out, M, N, K)

    out = out.view(inp.size(0), -1)
    out = out.transpose(0, dim).contiguous()

    if not keepdim:
        out = out.squeeze(dim)

    return out

@triton.jit
def argmax_kernel_1(
    inp,
    mid_value,
    mid_index,
    M,
    BLOCK_SIZE: tl.constexpr,
    INT64_INDEX: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inp_ptrs = inp + offset
    mid_value_ptrs = mid_value + pid
    mid_index_ptrs = mid_index + pid

    mask = offset < M

    inp_val = tl.load(inp_ptrs, mask=mask, other=-float("inf"))
    inp_idx = tl.where(mask, offset, -1)

    mid_val = tl.max(inp_val)
    mid_idx = tl.max(inp_idx)

    tl.store(mid_value_ptrs, mid_val)
    tl.store(mid_index_ptrs, mid_idx if INT64_INDEX else mid_idx.to(tl.int32))

@triton.jit
def argmax_kernel_2(mid_value, mid_index, out, mid_size, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mid_value_ptrs = mid_value + offset
    mid_index_ptrs = mid_index + offset

    mask = offset < mid_size

    mid_val = tl.load(mid_value_ptrs, mask=mask)
    mid_idx = tl.load(mid_index_ptrs, mask=mask)

    max_val = tl.max(mid_val)
    argmax = tl.max(mid_idx)

    tl.store(out, argmax)

@triton.jit
def argmax_kernel(
    inp,
    out,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    offset_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offset_n = pid_k * BLOCK_N + tl.arange(0, BLOCK_N)

    inp = inp + offset_m[:, None] * M + offset_n[None, :]
    out = out + offset_m[:, None] * N + offset_n[None, :]

    mask_m = offset_m < M
    mask_n = offset_n < N

    inp = tl.where(mask_m[:, None] & mask_n[None, :], inp, float("-inf"))

    out_val = tl.max(inp, axis=1)
    out_idx = tl.argmax(inp, axis=1)

    tl.store(out + offset_m[:, None], out_idx, mask=mask_m[:, None])
    tl.store(out + offset_m[:, None] + N, out_val, mask=mask_m[:, None])
