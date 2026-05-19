import triton
import triton.language as tl
import torch

@triton.jit
def _addmm_kernel(
    alpha, beta, IS_BETA_ZERO: tl.constexpr,
    M, N, K, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    A_ptr, B_ptr, C_ptr, D_ptr
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_warp = num_pid_m * num_pid_n
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    rk = tl.arange(0, BLOCK_SIZE_K)
    A = A_ptr + (rm[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B_ptr + (rk[:, None] * stride_bk + rn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(A, mask=k + rk[None, :] < K, other=0.0)
        b = tl.load(B, mask=k + rk[:, None] < K, other=0.0)
        accumulator += tl.dot(a, b)
        A += BLOCK_SIZE_K * stride_ak
        B += BLOCK_SIZE_K * stride_bk

    if IS_BETA_ZERO:
        accumulator = accumulator * alpha
    else:
        C = C_ptr + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
        c = tl.load(C, mask=rm[:, None] < M and rn[None, :] < N, other=0.0)
        accumulator = accumulator * alpha + c * beta

    D = D_ptr + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    tl.store(D, accumulator, mask=rm[:, None] < M and rn[None, :] < N)

import torch
import triton
import triton.language as tl

def addmm(input: torch.Tensor, mat1: torch.Tensor, mat2: torch.Tensor, *, beta=1.0, alpha=1.0, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    # Check input dimensions
    assert mat1.dim() == 2, "mat1 must be a 2D tensor"
    assert mat2.dim() == 2, "mat2 must be 2D tensor"
    assert mat1.size(1) == mat2.size(0), "Incompatible dimensions for matrix multiplication"
    
    # Determine output shape
    n, m = mat1.size()
    p = mat2.size(1)
    
    # Ensure input is broadcastable with (n, p)
    input = input.expand(n, p)
    
    # Determine output tensor
    if out is None:
        out = torch.empty((n, p), dtype=input.dtype, device=input.device)
    else:
        assert out.shape == (n, p), "Output tensor must have shape (n, p)"
        assert out.dtype == input.dtype, "Output tensor must have the same dtype as input"
        assert out.device == input.device, "Output tensor must be on the same device as input"
    
    # Define block sizes
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 16
    
    # Launch the Triton kernel
    grid = lambda META: (
        triton.cdiv(n, META['BLOCK_SIZE_M']) * triton.cdiv(p, META['BLOCK_SIZE_N']),
    )
    
    _addmm_kernel[grid](
        alpha, beta, beta == 0.0,
        n, p, m, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
        mat1.stride(0), mat1.stride(1),
        mat2.stride(0), mat2.stride(1),
        input.stride(0), input.stride(1),
        out.stride(0), out.stride(1),
        mat1, mat2, input, out
    )
    
    return out
