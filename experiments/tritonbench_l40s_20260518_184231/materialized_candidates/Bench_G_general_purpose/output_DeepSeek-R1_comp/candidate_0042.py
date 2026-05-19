import torch
import triton
import triton.language as tl
from typing import Optional

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_CS': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_CS': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_CS': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_CS': 16}, num_stages=5, num_warps=1),
    ],
    key=['K', 'N', 'CS'],
)
@triton.jit
def _bmm_chunk_bwd_kernel(
    a_ptr, dout_ptr, res_ptr, db_ptr,
    stride_a_batch, stride_a_csize, stride_a_k,
    stride_dout_batch, stride_dout_csize, stride_dout_m, stride_dout_n,
    stride_db_batch, stride_db_k, stride_db_n,
    Batch, CS, K, N,
    HAS_RESIDUAL: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_CS: tl.constexpr,
):
    batch_id = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    offs_k = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_cs = tl.arange(0, BLOCK_SIZE_CS)

    a_ptr += batch_id * stride_a_batch
    dout_ptr += batch_id * stride_dout_batch
    db_ptr += batch_id * stride_db_batch

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for cs in range(0, CS, BLOCK_SIZE_CS):
        a_chunk_ptr = a_ptr + (cs + offs_cs[:, None]) * stride_a_csize + offs_k[None, :] * stride_a_k
        do_chunk_ptr = dout_ptr + (cs + offs_cs[:, None]) * stride_dout_csize + offs_n[None, :] * stride_dout_n

        a = tl.load(a_chunk_ptr, mask=(cs + offs_cs[:, None] < CS) & (offs_k[None, :] < K), other=0.0)
        dout = tl.load(do_chunk_ptr, mask=(cs + offs_cs[:, None] < CS) & (offs_n[None, :] < N), other=0.0)

        acc += tl.dot(a, dout, allow_tf32=True)

    if HAS_RESIDUAL:
        res_ptr += batch_id * stride_db_batch
        res_ptrs = res_ptr + offs_k[:, None] * stride_db_k + offs_n[None, :] * stride_db_n
        res = tl.load(res_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc += res

    db_ptrs = db_ptr + offs_k[:, None] * stride_db_k + offs_n[None, :] * stride_db_n
    tl.store(db_ptrs, acc, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N))

def _bmm_chunk_bwd(
    a: torch.Tensor,
    dout: torch.Tensor,
    res: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    assert a.dim() == 3 and dout.dim() == 3, "Inputs must be 3D tensors"
    B, CS, K = a.shape
    B_do, CS_do, N = dout.shape
    assert B == B_do and CS == CS_do, "Batch and chunk size mismatch"

    a = a.contiguous()
    dout = dout.contiguous()
    db = torch.empty((B, K, N), device=a.device, dtype=a.dtype)

    HAS_RESIDUAL = res is not None
    if HAS_RESIDUAL:
        assert res.shape == (B, K, N), "Residual shape mismatch"
        res = res.contiguous()
    else:
        res = torch.empty(0, device=a.device)

    grid = (B, triton.cdiv(K, 128), triton.cdiv(N, 256))  # Default config grid for initialization
    def grid_fn(meta):
        return (B, triton.cdiv(K, meta['BLOCK_SIZE_M']), triton.cdiv(N, meta['BLOCK_SIZE_N']))

    with torch.cuda.device(a.device.index):
        _bmm_chunk_bwd_kernel[grid_fn](
            a, dout, res, db,
            a.stride(0), a.stride(1), a.stride(2),
            dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
            db.stride(0), db.stride(1), db.stride(2),
            B, CS, K, N,
            HAS_RESIDUAL,
        )
    return db
