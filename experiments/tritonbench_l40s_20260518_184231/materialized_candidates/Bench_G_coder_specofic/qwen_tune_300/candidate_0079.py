import torch
import triton
import triton.language as tl

@triton.jit
def _quantize_global_transpose(
    A,
    B,
    M,
    N,
    absmax_inv,
    stride_am,
    stride_an,
    stride_bm,
    stride_bn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    A = A + (rm[:, None] * stride_am + rn[None, :] * stride_an)
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    a = tl.load(A, mask=mask)
    b = tl.trans(a) / absmax_inv
    b = (b + 128).to(tl.int8)
    bm = pid_n * BLOCK_M + tl.arange(0, BLOCK_M)
    bn = pid_m * BLOCK_N + tl.arange(0, BLOCK_N)
    B = B + (bm[:, None] * stride_bm + bn[None, :] * stride_bn)
    mask = (bm < M)[:, None] & (bn < N)[None, :]
    tl.store(B, b, mask=mask)

def quantize_global_transpose(A: torch.Tensor):
    M, N = A.shape
    absmax = A.abs().max().unsqueeze(0)
    absmax_inv = absmax.reciprocal()
    B = torch.empty(N, M, device=A.device, dtype=torch.int8)
    def grid(meta): return (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]), )
    _quantize_global_transpose[grid](A, B, M, N, absmax_inv, stride_am=A.stride(0), stride_an=A.stride(1),
                                     stride_bm=B.stride(0), stride_bn=B.stride(1))
    return B
