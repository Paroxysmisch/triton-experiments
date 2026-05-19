import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.autograd.function import Function
from triton.common.backend import register_backend

class CudaBackend:
    stub_so_path = ""

def register_triton_cuda_backend():
    register_backend("cuda", CudaBackend)

def next_power_of_2(n):
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n += 1
    return n

def destindex_copy_kv(K: Tensor, DestLoc: Tensor, Out: Tensor):
    seq_len, head_num, head_dim = K.shape
    assert K.shape == DestLoc.shape
    assert K.shape == Out.shape

    BLOCK_HEAD = next_power_of_2(head_num)
    grid = (seq_len,)
    _fwd_kernel_destindex_copy_kv[grid](
        K,
        DestLoc,
        Out,
        K.stride(0),
        K.stride(1),
        K.stride(2),
        DestLoc.stride(0),
        DestLoc.stride(1),
        Out.stride(0),
        Out.stride(1),
        Out.stride(2),
        BLOCK_HEAD,
        num_warps=1,
        num_stages=1,
    )
    return Out

@triton.jit
def _fwd_kernel_destindex_copy_kv(
    K,
    DestLoc,
    Out,
    stride_k_bs,
    stride_k_h,
    stride_k_d,
    stride_dloc_bs,
    stride_dloc_h,
    stride_o_bs,
    stride_o_h,
    stride_o_d,
    BLOCK_HEAD,
    **meta,
):
    cur_index = tl.program_id(0)
    offs_d = tl.arange(0, meta["BLOCK_SIZE"])
    offs_h = tl.arange(0, BLOCK_HEAD)[:, None]

    k_ptrs = (
        K
        + cur_index * stride_k_bs
        + (offs_h * stride_k_h + offs_d[None, :] * stride_k_d)
        .to(tl.int64)
        .to(tl.int64)
    )
    o_ptrs = (
        Out
        + cur_index * stride_o_bs
        + (offs_h * stride_o_h + offs_d[None, :] * stride_o_d)
        .to(tl.int64)
        .to(tl.int64)
    )
    dloc_ptrs = (
        DestLoc
        + cur_index * stride_dloc_bs
        + (offs_h * stride_dloc_h + offs_d[None, :])
        .to(tl.int64)
        .to(tl.int64)
    )
    d_index = tl.load(dloc_ptrs, mask=offs_h, other=0)
    tl.store(
        o_ptrs,
        tl.load(k_ptrs, mask=offs_h[:, None], other=0.0),
        mask=d_index[:, None],
    )
    return
