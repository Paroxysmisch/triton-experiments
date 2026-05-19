import torch
import triton
import triton.language as tl

# Triton kernel for matrix multiplication
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 32, "NUM_STAGES": 4, "NUM_WARPS": 8}, num_stages=4, num_warps=8),
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "NUM_STAGES": 4, "NUM_WARPS": 4}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "NUM_STAGES": 4, "NUM_WARPS": 8}, num_stages=4, num_warps=8),
        triton.Config({"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "NUM_STAGES": 4, "NUM_WARPS": 8}, num_stages=4, num_warps=8),
        triton.Config({"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "NUM_STAGES": 4, "NUM_WARPS": 8}, num_stages=4, num_warps=8),
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 32, "NUM_STAGES": 4, "NUM_WARPS": 8}, num_stages=4, num_warps=8),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "NUM_STAGES": 4, "NUM_WARPS": 8}, num_stages=4, num_warps=8),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32, "NUM_STAGES": 4, "NUM_WARPS": 8}, num_stages=4, num_warps=8),
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 16, "BLOCK_SIZE_K": 32, "NUM_STAGES": 2, "NUM_WARPS": 4}, num_stages=2, num_warps=4),
        # Additional configurations...
    ],
    key=["a_row_stride", "a_col_stride", "b_row_stride", "b_col_stride", "N", "K", "M"],
)
@triton.jit
def matmul_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    # Matrix dimensions
    M,
    N,
    K,
    a_row_stride,
    a_col_stride,
    b_row_stride,
    b_col_stride,
    c_row_stride,
    c_col_stride,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    # Data Types
    T_ACC,
):

    # ---------------
    # Kernel function
    # ---------------

    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    block_offset_m = pid_m * BLOCK_SIZE_M
    block_offset_n = pid_n * BLOCK_SIZE_N

    offs_am = block_offset_m * a_row_stride + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = block_offset_n * b_col_stride + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * a_row_stride + offs_k[None, :] * a_col_stride)  # (BLOCK_SIZE_M, BLOCK_SIZE_K)
    b_ptrs = b_ptr + (offs_k[:, None] * b_row_stride + offs_bn[None, :] * b_col_stride)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=T_ACC)

    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k, other=0.0)  # (BLOCK_SIZE_M, BLOCK_SIZE_K)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k, other=0.0)  # (BLOCK_SIZE_K, BLOCK_SIZE_N)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * a_col_stride
        b_ptrs += BLOCK_SIZE_K * b_row_stride

    c = _fp8_to_fp16(accumulator) if T_ACC == tl.float8e5m2 else accumulator

    offs_cm = block_offset_m + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = block_offset_n + tl.arange(0, BLOCK_SIZE_N)
    mask = (offs_cm < M)[:, None] & (offs_cn < N)[None, :]
    c_ptrs = c_ptr + (offs_cm * c_row_stride + offs_cn * c_col_stride)
    tl.store(c_ptrs, c, mask=mask)

# Function to call the Triton kernel
def matmul(a, b, *, beta=1.0, alpha=1.0, out=None, group_size_m=None):
    
    
    # ... 
    # Function implementation to call the Triton kernel
    
    
    def grid(meta):
        return (triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(N, meta["BLOCK_SIZE_N"]), )
    
    matmul_kernel[grid](
        a,
        b,
        out,
        M,
        N,
        K,
        stride(a),
        stride(a, 0, 1),
        stride(b),
        stride(b, 0, 1),
        stride(out),
        stride(out, 0, 1),
