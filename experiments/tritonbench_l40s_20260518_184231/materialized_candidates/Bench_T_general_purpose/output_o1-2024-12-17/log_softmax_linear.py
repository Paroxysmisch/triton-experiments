import torch
import triton
import triton.language as tl

# Kernel: matrix multiply + optional bias
@triton.jit
def _matmul_bias_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Rows of x and y processed by this program
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # Cols of w and y processed by this program
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # K dimension
    offs_k = tl.arange(0, BLOCK_K)
    
    # Create pointers for x and y
    x_ptrs = x_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    w_ptrs = w_ptr + (offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk)
    
    # Initialize accumulators
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Loop over K
    for k in range(0, K, BLOCK_K):
        # Load a block of x
        x = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & (k + offs_k[None, :] < K), other=0.0)
        # Load a block of w
        w = tl.load(w_ptrs, mask=(offs_n[None, :] < N) & (k + offs_k[:, None] < K), other=0
