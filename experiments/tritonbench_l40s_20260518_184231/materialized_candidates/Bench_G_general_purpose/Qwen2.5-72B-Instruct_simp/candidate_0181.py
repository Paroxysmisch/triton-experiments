import triton
import triton.language as tl

@triton.jit
def rms_matmul_rbe(
    X, RMS_W, W, C, 
    X_row_stride, X_col_stride, 
    RMS_W_row_stride, RMS_W_col_stride, 
    W_row_stride, W_col_stride, 
    C_row_stride, C_col_stride, 
    M, N, K, 
    rotary_emb, 
    rotary_dim, 
    BLOCK_M: tl.constexpr, 
    BLOCK_N: tl.constexpr, 
    BLOCK_K: tl.constexpr, 
    PRECISION: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = num_pid_m * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * num_pid_m
    group_size = num_pid_in_group
    pid_m = first_pid_m + (pid % num_pid_m)
    pid_n = (pid % group_size) // num_pid_m

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    X = X + (rm[:, None] * X_row_stride + rk[None, :] * X_col_stride)
    RMS_W = RMS_W + (rm[:, None] * RMS_W_row_stride + rk[None, :] * RMS_W_col_stride)
    W = W + (rk[:, None] * W_row_stride + rn[None, :] * W_col_stride)
    C = C + (rm[:, None] * C_row_stride + rn[None, :] * C_col_stride)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=PRECISION)

    for k in range(0, K, BLOCK_K):
        x = tl.load(X)
        rms_w = tl.load(RMS_W)
        w = tl.load(W)
        
        # RMS normalization
        rms_x = tl.sqrt(tl.sum(x * x, axis=1) / K)
        x = x / rms_x[:, None]
        
        # Apply rotary embeddings if enabled
        if rotary_emb:
            rotary_mask = tl.arange(0, BLOCK_K) < rotary_dim
            x = tl.where(rotary_mask, x * tl.cos(rotary_emb) - x * tl.sin(rotary_emb), x)
        
        # Matrix multiplication
        acc += tl.dot(x, w)

        X += BLOCK_K * X_col_stride
        RMS_W += BLOCK_K * RMS_W_col_stride
        W += BLOCK_K * W_col_stride

    tl.store(C, acc)

import torch
import triton
import triton.runtime as trt

def rms_matmul_rbe_qkv_wrapper(X, RMS_W, W, rotary_emb=None, rotary_dim=None):
    M, K = X.shape
    _, N = W.shape

    # Allocate output tensor
    C = torch.empty((M, N), dtype=X.dtype, device=X.device)

    # Define block sizes
    BLOCK_M = 16
    BLOCK_N = 16
    BLOCK_K = 16

    # Define grid size
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)

    # Launch kernel
    rms_matmul_rbe[grid](
        X, RMS_W, W, C, 
        X.stride(0), X.stride(1), 
        RMS_W.stride(0), RMS_W.stride(1), 
        W.stride(0), W.stride(1), 
        C.stride(0), C.stride(1), 
        M, N, K, 
        rotary_emb, 
        rotary_dim, 
        BLOCK_M, BLOCK_N, BLOCK_K, 
        PRECISION=tl.float16 if X.dtype == torch.float16 else tl.float32
    )

    return C
