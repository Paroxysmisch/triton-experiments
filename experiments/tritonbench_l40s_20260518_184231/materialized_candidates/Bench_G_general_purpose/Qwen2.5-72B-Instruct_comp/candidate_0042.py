import triton
import triton.language as tl

@triton.jit
def _bmm_chunk_bwd_kernel(
    a_ptr, dout_ptr, res_ptr, db_ptr,
    stride_a_batch, stride_a_m, stride_a_k,
    stride_dout_batch, stride_dout_m, stride_dout_csize_m,
    stride_db_batch, stride_db_k, stride_db_n,
    stride_res_batch, stride_res_m, stride_res_n,
    M, N, K, CSIZE,
    HAS_RESIDUAL: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_CS: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_batch = num_pid_m * num_pid_n
    batch_idx = pid // num_pid_in_batch
    pid_mn = pid % num_pid_in_batch
    pid_m = pid_mn // num_pid_n
    pid_n = pid_mn % num_pid_n

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_CS)

    a_batch_ptr = a_ptr + batch_idx * stride_a_batch
    dout_batch_ptr = dout_ptr + batch_idx * stride_dout_batch
    db_batch_ptr = db_ptr + batch_idx * stride_db_batch
    res_batch_ptr = res_ptr + batch_idx * stride_res_batch if HAS_RESIDUAL else 0

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for c in range(0, CSIZE, BLOCK_SIZE_CS):
        a = tl.load(a_batch_ptr + offs_m[:, None] * stride_a_m + (c + offs_k)[None, :] * stride_a_k)
        dout = tl.load(dout_batch_ptr + (c + offs_k)[:, None] * stride_dout_csize_m + offs_n[None, :] * stride_dout_n)
        acc += tl.dot(a, dout)

    if HAS_RESIDUAL:
        res = tl.load(res_batch_ptr + offs_m[:, None] * stride_res_m + offs_n[None, :] * stride_res_n)
        acc += res

    tl.store(db_batch_ptr + offs_m[:, None] * stride_db_k + offs_n[None, :] * stride_db_n, acc)

import torch
import triton
import triton.language as tl

def _bmm_chunk_bwd(a, dout, res=None, db=None):
    assert a.is_contiguous(), "Input tensor 'a' must be contiguous."
    assert dout.is_contiguous(), "Input tensor 'dout' must be contiguous."
    if res is not None:
        assert res.is_contiguous(), "Input tensor 'res' must be contiguous."
    if db is not None:
        assert db.is_contiguous(), "Output tensor 'db' must be contiguous."

    B, M, K = a.shape
    B, CSIZE, N = dout.shape
    if res is not None:
        B, M, N = res.shape
    if db is None:
        db = torch.zeros((B, K, N), device=a.device, dtype=a.dtype)

    grid = (triton.cdiv(M, 16) * triton.cdiv(N, 16) * B,)

    _bmm_chunk_bwd_kernel[grid](
        a, dout, res, db,
        a.stride(0), a.stride(1), a.stride(2),
        dout.stride(0), dout.stride(1), dout.stride(2),
        db.stride(0), db.stride(1), db.stride(2),
        res.stride(0), res.stride(1), res.stride(2) if res is not None else 0,
        M, N, K, CSIZE,
        res is not None,
        BLOCK_SIZE_M=16, BLOCK_SIZE_N=16, BLOCK_SIZE_CS=16
    )

    return db
