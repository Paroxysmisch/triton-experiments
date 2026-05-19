import triton
import triton.language as tl

@triton.jit
def _bmm_chunk_bwd_kernel(
    a_ptr, dout_ptr, db_ptr, res_ptr,  # Pointers to input/output data
    M, N, K, CS,  # Dimensions
    stride_am, stride_ak,  # Strides for matrix a
    stride_dout_bk, stride_dout_bn,  # Strides for matrix dout
    stride_db_km, stride_db_kn,  # Strides for matrix db
    stride_res_m,  # Stride for residual matrix
    BLOCK_SIZE_M: tl.constexpr,  # Block size for M dimension
    BLOCK_SIZE_N: tl.constexpr,  # Block size for N dimension
    BLOCK_SIZE_CS: tl.constexpr,  # Block size for CS dimension
    HAS_RESIDUAL: tl.constexpr  # Flag to indicate if there is a residual term
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_batch = num_pid_m * num_pid_n
    batch_id = pid // num_pid_in_batch
    pid_mn = pid % num_pid_in_batch
    pid_m = pid_mn // num_pid_n
    pid_n = pid_mn % num_pid_n

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_CS)

    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    dout_ptrs = dout_ptr + (offs_k[:, None] * stride_dout_bk + offs_n[None, :] * stride_dout_bn)
    db_ptrs = db_ptr + (offs_m[:, None] * stride_db_km + offs_n[None, :] * stride_db_kn)

    a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
    dout_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)

    a = tl.load(a_ptrs, mask=a_mask, other=0.0)
    dout = tl.load(dout_ptrs, mask=dout_mask, other=0.0)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_CS):
        db = tl.dot(a, dout)
        acc += db

    db_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.atomic_add(db_ptrs, acc, mask=db_mask)

    if HAS_RESIDUAL:
        res_ptrs = res_ptr + (offs_m * stride_res_m)
        res = tl.load(res_ptrs, mask=offs_m < M, other=0.0)
        db += res
        tl.store(db_ptrs, db, mask=db_mask)

import torch
import triton
import triton.language as tl

def _bmm_chunk_bwd(a, dout, db, res=None, BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_CS=32):
    # Ensure the inputs are contiguous
    a = a.contiguous()
    dout = dout.contiguous()
    db = db.contiguous()
    if res is not None:
        res = res.contiguous()

    # Get dimensions
    B, M, K = a.shape
    B, K, N = dout.shape
    B, M, N = db.shape

    # Get strides
    stride_am = a.stride(1)
    stride_ak = a.stride(2)
    stride_dout_bk = dout.stride(1)
    stride_dout_bn = dout.stride(2)
    stride_db_km = db.stride(1)
    stride_db_kn = db.stride(2)
    stride_res_m = res.stride(1) if res is not None else 0

    # Launch the kernel
    grid = (B * triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),)
    _bmm_chunk_bwd_kernel[grid](
        a, dout, db, res,
        M, N, K, CS,
        stride_am, stride_ak,
        stride_dout_bk, stride_dout_bn,
        stride_db_km, stride_db_kn,
        stride_res_m,
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_CS,
        res is not None
    )

# Example usage
B, M, K, N = 2, 1024, 1024, 1024
a = torch.randn((B, M, K), device='cuda')
dout = torch.randn((B, K, N), device='cuda')
db = torch.zeros((B, M, N), device='cuda')
res = torch.randn((B, M), device='cuda')

_bmm_chunk_bwd(a, dout, db, res)
