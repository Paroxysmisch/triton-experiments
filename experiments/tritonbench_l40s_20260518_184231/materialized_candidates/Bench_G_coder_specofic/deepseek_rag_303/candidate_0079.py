import triton
import triton.language as tl
import torch
from torch._inductor.triton_heuristics import grid

@triton.jit
def _quantize_global_transpose(
        A, stride_am, stride_an, B, stride_bm, stride_bn,
        M, N, absmax, absmax_inv, GROUP_M: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    pid_m = pid // N
    pid_n = pid % N

    group_id = (pid_m * GROUP_M + (pid_n // BLOCK_N))
    mask_groups = (group_id == (pid_m * GROUP_M + (pid_n // BLOCK_N)))

    offs_m = group_id * BLOCK_M + (tl.arange(0, BLOCK_M) + tl.where(mask_groups, 0, (1 + pid_m % GROUP_M) * BLOCK_M)) % M
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    a_ptr = A + (offs_m[:, None] * stride_am + offs_n[None, :] * stride_an)
    b_ptr = B + (offs_m[:, None] * stride_bm + offs_n[None, :] * stride_bn)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    a = tl.load(a_ptr, mask=mask, other=0.0).to(tl.float32)
    q = a * absmax_inv
    q = tl.where(q > 127, 127, q)
    q = tl.where(q < -128, -128, q)
    b = q.to(tl.int8)
    tl.store(b_ptr, b, mask=mask)

def quantize_global_transpose(A: torch.Tensor, B: torch.Tensor):
    assert A.is_cuda and B.is_cuda
    M, N = A.shape
    absmax = A.abs().max().to(A.dtype)
    absmax_inv = 127.0 / absmax
    GROUP_M = 128
    BLOCK_M = 128
    BLOCK_N = 16

    def grid(META):
        return [triton.cdiv(M * N, META['BLOCK_M'] * META['BLOCK_N'])]
    _quantize_global_transpose[grid](A, B, M, N, absmax, absmax_inv, GROUP_M, BLOCK_M, BLOCK_N)
