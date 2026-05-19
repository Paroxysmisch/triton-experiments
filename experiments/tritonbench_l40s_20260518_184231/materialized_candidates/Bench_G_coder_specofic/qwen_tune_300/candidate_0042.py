import torch
import triton
import triton.language as tl
from torch import Tensor
from typing import Optional

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_CS": 64, "NUM_STAGES": 3, "NUM_WARPS": 8}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_CS": 32, "NUM_STAGES": 4, "NUM_WARPS": 4}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_CS": 32, "NUM_STAGES": 4, "NUM_WARPS": 4}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_CS": 32, "NUM_STAGES": 4, "NUM_WARPS": 4}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_CS": 32, "NUM_STAGES": 4, "NUM_WARPS": 4}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_CS": 32, "NUM_STAGES": 4, "NUM_WARPS": 4}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_CS": 32, "NUM_STAGES": 5, "NUM_WARPS": 2}, num_stages=5, num_warps=2),
        triton.Config({"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_CS": 32, "NUM_STAGES": 5, "NUM_WARPS": 2}, num_stages=5, num_warps=2),
    ],
    key=["chunk_size", "K", "IS_CAUSAL"],
)
@triton.jit
def _bmm_chunk_bwd_kernel(
    a_ptr, dout_ptr, db_ptr, dc_ptr, res_ptr,
    batch, csize, chunk_size, K, ngroups,
    stride_a_batch, stride_a_csize, stride_a_k,
    stride_dout_batch, stride_dout_csize_m, stride_dout_csize_n, stride_dout_k,
    stride_db_batch, stride_db_chunk, stride_db_head, stride_db_csize,
    stride_dc_batch, stride_dc_chunk, stride_dc_head, stride_dc_k,
    stride_res_batch, stride_res_csize, stride_res_k,
    IS_CAUSAL: tl.constexpr,
    dot_dtype: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_CS: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_batch = tl.program_id(axis=0)
    pid_chunk = tl.program_id(axis=1)
    pid_head = tl.program_id(axis=2)
    a_ptr += pid_batch * stride_a_batch + pid_chunk * chunk_size * stride_a_csize
    dout_ptr += pid_batch * stride_dout_batch + pid_chunk * chunk_size * stride_dout_csize_m

    offs_m = tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    db_ptr += pid_batch * stride_db_batch + pid_chunk * stride_db_chunk + pid_head * stride_db_head
    dc_ptr += pid_batch * stride_dc_batch + pid_chunk * stride_dc_chunk + pid_head * stride_dc_head
    aux = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    chunk_size_limit = min(chunk_size, csize - pid_chunk * chunk_size)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        if IS_CAUSAL:
            k_end = min(chunk_size_limit, (k + 1) * BLOCK_SIZE_K)
            k_start = max(k * BLOCK_SIZE_K, k_end - BLOCK_SIZE_K)
            _offs_k = k_start + offs_k
            a_ptrs = a_ptr + (offs_m[:, None] * stride_a_csize + _offs_k[None, :] * stride_a_k)
            a_mask = (_offs_k[None, :] < K) & (offs_m[:, None] < chunk_size_limit)
            a = tl.load(a_ptrs, mask=a_mask, other=0.0)
            offs_k_load = _offs_k
        else:
            k_start = k * BLOCK_SIZE_K
            k_end = min(K, k_start + BLOCK_SIZE_K)
            _offs_k = offs_k + k_start
            a_ptrs = a_ptr + (offs_m[:, None] * stride_a_csize + _offs_k[None, :] * stride_a_k)
            a_mask = (_offs_k[None, :] < k_end) & (offs_m[:, None] < chunk_size_limit)
            a = tl.load(a_ptrs, mask=a_mask, other=0.0)
            offs_k_load = _offs_k

        if HAS_RESIDUAL:
            res_ptrs = res_ptr + (offs_m[:, None] * stride_res_csize + offs_n[None, :] * stride_res_k)
            res_mask = (offs_n[None, :] < BLOCK_SIZE_N) & (offs_m[:, None] < chunk_size_limit)
            res = tl.load(res_ptrs, mask=res_mask, other=0.0)
            aux += res.to(dot_dtype)

        dout_ptrs = dout_ptr + (offs_m[:, None] * stride_dout_csize_m + offs_n[None, :] * stride_dout_csize_n)
        dout_mask = (offs_n[None, :] < chunk_size_limit) & (offs_m[:, None] < chunk_size_limit)
        dout = tl.load(dout_ptrs, mask=dout_mask, other=0.0).to(dot_dtype)

        aux = tl.dot(a, dout, acc=aux)
        a = a.to(dot_dtype)

        if IS_CAUSAL:
            aux_mask = (offs_m[:, None] < k_end) & (offs_n[None, :] < k_end)
        else:
            aux_mask = (offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < chunk_size_limit)

        aux = tl.where(aux_mask, aux, 0.0)
        dc = tl.sum(aux, axis=1)

        if IS_CAUSAL:
            offs_m_db = offs_m + pid_chunk * chunk_size
            offs_n_db = offs_k_load
        else:
            offs_m_db = offs_m + pid_chunk * chunk_size
            offs_n_db = offs_n

        db_ptrs = db_ptr + offs_m_db * stride_db_csize + offs_n_db * stride_db_cs
        tl.store(db_ptrs, aux.to(db_ptr.dtype.element_ty), mask=aux_mask)

        offs_m_dc = offs_n
        offs_n_dc = offs_k_load

        dc_ptrs = dc_ptr + offs_m_dc[:, None] * stride_dc_k + offs_n_dc[None, :] * stride_dc_batch
        tl.store(dc_ptrs, dc[:, None].to(dc_ptr.dtype.element_ty), mask=aux_mask)

    return


def _bmm_chunk_bwd(
    a: Tensor, dout: Tensor, res: Optional[Tensor],
    chunk_size: int, ngroups: int, residual: bool, is_causal: bool
) -> tuple[Tensor, Tensor]:
    assert a.shape[-2:] == (dout.shape[-2], dout.shape[-1])
    assert a.stride(-1) == 1
    batch, csize, K = a.shape
    assert dout.shape == (batch, csize, csize, K)
    assert dout.stride(-1) == 1
    if res is not None:
        assert res.shape == (batch, csize, K)
        assert res.stride(-1) == 1
    db = a.new_empty(batch, ngroups, csize, chunk_size)
    dc = a.new_empty(batch, ngroups, chunk_size, K)
    dot_dtype = tl.bfloat16 if a.dtype == torch.bfloat16 else tl.float32

    def grid(meta): return (batch, triton.cdiv(chunk_size, meta["BLOCK_SIZE_CS"]), ngroups)
    _bmm_chunk_bwd_kernel[grid](
        a, dout, db, dc, res if residual else a,
        batch, csize, chunk_size, K, ngroups,
        a.stride(0), a.stride(1), a.stride(2),
        dout.stride(0), dout.stride(1), dout.stride(3), dout.stride(2),
        db.stride(0), db.stride(2), db.stride(1), db.stride(3),
        dc.stride(0), dc.stride(2), dc.stride(1), dc.stride(3),
        res.stride(0) if residual else 0,
