import triton
import triton.language as tl
import torch
from torch import Tensor
from torch.autograd.function import Function
from torch.autograd import register_function

@triton.jit
def mv_kernel(
    A, stride_am, stride_ak,
    B, stride_bk,
    C, stride_ck,
    M, N,
    BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr
):
    pid = tl.program_id(0)
    offset_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offset_n = tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(offset_m % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(offset_n % N, BLOCK_N), BLOCK_N)

    A = A + (ram[:, None] * stride_am + rbn[None, :] * 1)
    B = B + offset_n * stride_bk
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(N, BLOCK_N)):
        b_b = tl.load(B)
        b_a = tl.load(A, mask=offset_m[:, None] < M, other=0.0)
        acc += tl.dot(b_a, b_b)
        A += BLOCK_N * stride_ak
        B += BLOCK_N * stride_bk
    acc = tl.trans(acc)
    C = C + offset_n[None, :] * stride_ck
    tl.store(C, acc, mask=offset_m[:, None] < M)

def mv(inp, vec):
    assert inp.shape[1] == vec.shape[0], "incompatible dimensions"
    M, N = inp.shape
    out = torch.empty((M,), device=inp.device, dtype=inp.dtype)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']), )
    with torch.cuda.device(inp.device):
        mv_kernel[grid](
            inp, inp.stride(0), inp.stride(1),
            vec, vec.stride(0),
            out, out.stride(0),
            M, N)
    return out
