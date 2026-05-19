import torch
import triton
import triton.language as tl
from torch import Tensor

@triton.jit
def max_kernel_1(in_ptr0, mid_ptr0, n: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n
    in_ptrs = in_ptr0 + offset
    in_vals = tl.load(in_ptrs, mask=mask, other=-float("inf"))
    mid = tl.max(in_vals)
    mid_ptrs = mid_ptr0 + pid
    tl.store(mid_ptrs, mid)

@triton.jit
def max_kernel_2(mid_ptr0, out_ptr0, n: tl.constexpr):
    offset = tl.arange(0, 1)
    mask = offset < 1
    mid_ptrs = mid_ptr0 + offset
    mid_vals = tl.load(mid_ptrs, mask=mask, other=-float("inf"))
    out = tl.max(mid_vals)
    out_ptrs = out_ptr0 + offset
    tl.store(out_ptrs, out)

def max(A: Tensor, dim=None, keepdim=False):
    if dim is None or keepdim:
        B = A.flatten()
        N = B.numel()
        block_size = triton.next_power_of_2(math.ceil(math.sqrt(N)))
        mid = torch.empty((triton.cdiv(N, block_size),), dtype=A.dtype, device=A.device)
        out = torch.empty((1,), dtype=A.dtype, device=A.device)
        with torch.cuda.device(A.device):
            max_kernel_1[(triton.cdiv(N, block_size), 1, 1)](B, mid, N, block_size)
            max_kernel_2[(1, 1, 1)](mid, out, N)
        if not keepdim:
            out = out.squeeze()
        return out
    else:
        M = A.size(dim)
        K = A.numel() // M
        A = A.contiguous()
        out_shape = list(A.shape)
        out_shape[dim] = 1
        out = torch.empty(out_shape, dtype=A.dtype, device=A.device)
        mid = torch.empty((A.size(0), 1, 1), dtype=A.dtype, device=A.device)
        grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), 1, 1)
        with torch.cuda.device(A.device):
            max_kernel[grid](A, out, M, K, dim, BLOCK_M=triton.next_power_of_2(math.ceil(math.sqrt(M)))
            )
        return out

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 1}),
        triton.Config({"BLOCK_M": 2}),
        triton.Config({"BLOCK_M": 4}),
        triton.Config({"BLOCK_M": 8}),
        triton.Config({"BLOCK_M": 16}),
        triton.Config({"BLOCK_M": 32}),
        triton.Config({"BLOCK_M": 64}),
    ],
    key=["K", "M"],
)
@triton.jit
def max_kernel(A, out, M, K, dim, BLOCK_M: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m_offset < M
    a_ptr = A + m_offset * K + pid_k
    a = tl.load(a_ptr, mask=m_mask & (pid_k < K), other=-float("inf"))
    out_ptr = out + m_offset * K + pid_k
    if dim == 0:
        out_val = tl.max(a)
        tl.store(out_ptr, out_val, mask=(pid_k < K))
    else:
        out_val = tl.max(a)
        tl.store(out_ptr, out_val, mask=m_mask & (pid_k < K))
