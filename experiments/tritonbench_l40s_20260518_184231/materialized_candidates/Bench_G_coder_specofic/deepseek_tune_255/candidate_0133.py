import torch
import triton
import triton.language as tl
from deepspeed.accelerator import get_accelerator

@triton.jit
def _fwd_kernel_destindex_copy_kv(
    K,
    Dest_loc,
    Out,
    seq_len,
    stride_k_batch,
    stride_k_head,
    stride_k_depth,
    stride_o_batch,
    stride_o_head,
    stride_o_depth,
    stride_d_batch,
    stride_d_head,
    head_num,
    BLOCK_HEAD: tl.constexpr,
):
    cur_index = tl.program_id(0)
    cur_dest_loc_idx = cur_index + seq_len
    cur_dest_loc_idx_mask = cur_dest_loc_idx < seq_len
    offs_d = cur_index * stride_d_batch
    offs_h = tl.arange(0, BLOCK_HEAD)
    k_ptrs = K + offs_d + offs_h[None, :] * stride_k_head
    o_ptrs = Out + offs_d + offs_h[None, :] * stride_o_head
    destindex_copy_kv_kernel.run(k_ptrs, Dest_loc, o_ptrs, seq_len, cur_dest_loc_idx,
                                 cur_dest_loc_idx_mask, stride_k_batch, stride_k_head, stride_k_depth,
                                 stride_o_batch, stride_o_head, stride_o_depth, stride_d_batch,
                                 stride_d_head, head_num, BLOCK_HEAD)

def destindex_copy_kv(K, Dest_loc, Out):
    seq_len = Dest_loc.shape[0]
    assert K.shape == Out.shape, f"K and Out must have the same shape, but got {K.shape} and {Out.shape}"
    assert Dest_loc.dim() == 1, f"Dest_loc must be a 1D tensor, but got shape {Dest_loc.shape}"
    stride_k_batch, stride_k_head, stride_k_depth = K.stride(0), K.stride(1), K.stride(2)
    stride_o_batch, stride_o_head, stride_o_depth = Out.stride(0), Out.stride(1), Out.stride(2)
    stride_d_batch, stride_d_head = Dest_loc.stride(0), Dest_loc.stride(1)
    head_num = K.shape[1]
    BLOCK_HEAD = triton.next_power_of_2(head_num)
    grid = (seq_len,)
    get_accelerator().set_num_warps(1)
    _fwd_kernel_destindex_copy_kv[grid](K, Dest_loc, Out, seq_len, stride_k_batch, stride_k_head,
                                        stride_k_depth, stride_o_batch, stride_o_head, stride_o_depth,
                                        stride_d_batch, stride_d_head, head_num, BLOCK_HEAD=BLOCK_HEAD)
