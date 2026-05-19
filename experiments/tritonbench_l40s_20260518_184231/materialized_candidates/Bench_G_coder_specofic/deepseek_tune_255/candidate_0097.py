import torch
import triton
import triton.language as tl
from torch.cuda.amp import custom_bwd, custom_fwd

@triton.jit
def _fwd_kernel_destindex_copy_kv(
    K,
    Out,
    DestLoc,
    seq_len,
    k_h_dim,
    k_d_dim,
    stride_k_b,
    stride_k_h,
    stride_k_d,
    stride_o_b,
    stride_o_h,
    stride_o_d,
    BLOCK_HEAD: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    cur_index = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_d = tl.program_id(2)

    cur_destloc = tl.load(DestLoc + cur_index)
    cur_k = tl.load(
        K + cur_destloc * stride_k_b + cur_head * stride_k_h + start_d * stride_k_d
    )
    tl.store(
        Out + cur_index * stride_o_b + cur_head * stride_o_h + start_d * stride_o_d,
        cur_k,
    )

def destindex_copy_kv(K, DestLoc, Out=None):
    batch_size, head_num, head_dim = K.shape
    assert K.shape == (batch_size, head_num, head_dim)
    if Out is None:
        Out = torch.empty_like(K)
    assert Out.shape == K.shape
    assert DestLoc.shape == (batch_size,)
    seq_len = batch_size

    BLOCK_HEAD = triton.next_power_of_2(head_num)
    BLOCK_DMODEL = triton.next_power_of_2(head_dim)
    grid = (seq_len, head_num, triton.cdiv(head_dim, BLOCK_DMODEL))

    _fwd_kernel_destindex_copy_kv[grid](
        K,
        Out,
        DestLoc,
        seq_len,
        head_num,
        head_dim,
        K.stride(0),
        K.stride(1),
        K.stride(2),
        Out.stride(0),
        Out.stride(1),
        Out.stride(2),
        BLOCK_HEAD,
        BLOCK_DMODEL,
    )
    return Out
