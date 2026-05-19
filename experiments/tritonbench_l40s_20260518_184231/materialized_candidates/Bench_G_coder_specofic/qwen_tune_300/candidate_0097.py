import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_destindex_copy_kv(
    K, DestLoc, Out,
    stride_k_bs, stride_k_h, stride_k_d,
    stride_o_bs, stride_o_h, stride_o_d,
    cur_index,
    BLOCK_HEAD: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr
):
    cur_index = cur_index + tl.program_id(0)
    offs_h = tl.arange(0, BLOCK_HEAD)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    dest_index = tl.load(DestLoc + cur_index)
    k_ptrs = K + cur_index * stride_k_bs + stride_k_h * offs_h[:, None] + stride_k_d * offs_d[None, :]
    o_ptrs = Out + dest_index * stride_o_bs + stride_o_h * offs_h[:, None] + stride_o_d * offs_d[None, :]
    k = tl.load(k_ptrs, mask=offs_h[:, None] < K.shape[1], other=0.0)
    tl.store(o_ptrs, k, mask=offs_h[:, None] < K.shape[1])
    return

@torch.no_grad()
def destindex_copy_kv(K, DestLoc, Out):
    seq_len = DestLoc.shape[0]
    assert K.shape[1] == Out.shape[1], "K and Out must have the same head number."
    assert K.shape[2] == Out.shape[2], "K and Out must have the same head dimension."
    BLOCK_HEAD = triton.next_power_of_2(K.shape[1])
    BLOCK_DMODEL = triton.next_power_of_2(K.shape[2])
    grid = (seq_len,)
    num_warps = 1
    _fwd_kernel_destindex_copy_kv[grid](
        K, DestLoc, Out,
        K.stride(0), K.stride(1), K.stride(2),
        Out.stride(0), Out.stride(1), Out.stride(2),
        cur_index=0,
        BLOCK_HEAD=BLOCK_HEAD,
        BLOCK_DMODEL=BLOCK_DMODEL,
        num_warps=num_warps,
        num_stages=1,
    )
    return
