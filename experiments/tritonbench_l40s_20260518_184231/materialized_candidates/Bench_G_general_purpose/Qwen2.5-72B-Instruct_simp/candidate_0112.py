import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_destindex_copy_kv(
    KV_nope_ptr, KV_rope_ptr, O_nope_ptr, O_rope_ptr, DestLoc_ptr,
    batch_size, seq_len, head_num, dim, stride_kv_nope_b, stride_kv_nope_h, stride_kv_nope_d,
    stride_kv_rope_b, stride_kv_rope_h, stride_kv_rope_d,
    stride_o_nope_b, stride_o_nope_h, stride_o_nope_d,
    stride_o_rope_b, stride_o_rope_h, stride_o_rope_d,
    stride_destloc_b, stride_destloc_s,
    BLOCK_SIZE_SEQ: tl.constexpr, BLOCK_SIZE_DIM: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(seq_len, BLOCK_SIZE_SEQ)
    num_pid_n = tl.cdiv(dim, BLOCK_SIZE_DIM)
    num_pid_in_batch = num_pid_m * num_pid_n
    batch_idx = pid // num_pid_in_batch
    pid = pid % num_pid_in_batch
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_SIZE_SEQ + tl.arange(0, BLOCK_SIZE_SEQ)
    offs_n = pid_n * BLOCK_SIZE_DIM + tl.arange(0, BLOCK_SEDIM_DIM)
    offs_b = batch_idx * stride_kv_nope_b

    KV_nope_block_ptr = KV_nope_ptr + offs_b
    KV_rope_block_ptr = KV_rope_ptr + offs_b
    O_nope_block_ptr = O_nope_ptr + offs_b
    O_rope_block_ptr = O_rope_ptr + offs_b
    DestLoc_block_ptr = DestLoc_ptr + batch_idx * stride_destloc_b

    for head_idx in range(head_num):
        kv_nope_head_ptr = KV_nope_block_ptr + head_idx * stride_kv_nope_h
        kv_rope_head_ptr = KV_rope_block_ptr + head_idx * stride_kv_rope_h
        o_nope_head_ptr = O_nope_block_ptr + head_idx * stride_o_nope_h
        o_rope_head_ptr = O_rope_block_ptr + head_idx * stride_o_rope_h

        dest_loc_head_ptr = DestLoc_block_ptr + head_idx * stride_destloc_s

        dest_loc = tl.load(dest_loc_head_ptr + offs_m, mask=offs_m < seq_len, other=0)
        dest_loc = dest_loc * stride_o_nope_d

        kv_nope_val = tl.load(kv_nope_head_ptr + offs_m[:, None] * stride_kv_nope_d + offs_n[None, :], mask=offs_m[:, None] < seq_len, other=0)
        kv_rope_val = tl.load(kv_rope_head_ptr + offs_m[:, None] * stride_kv_rope_d + offs_n[None, :], mask=offs_m[:, None] < seq_len, other=0)

        tl.store(o_nope_head_ptr + dest_loc[:, None] + offs_n[None, :], kv_nope_val, mask=offs_m[:, None] < seq_len)
        tl.store(o_rope_head_ptr + dest_loc[:, None] + offs_n[None, :], kv_rope_val, mask=offs_m[:, None] < seq_len)

import torch

def destindex_copy_kv(KV_nope, KV_rope, O_nope, O_rope, DestLoc):
    batch_size, seq_len, head_num, dim = KV_nope.shape
    assert KV_nope.shape == KV_rope.shape == O_nope.shape == O_rope.shape
    assert DestLoc.shape == (batch_size, seq_len, head_num)

    stride_kv_nope_b = KV_nope.stride(0)
    stride_kv_nope_h = KV_nope.stride(1)
    stride_kv_nope_d = KV_nope.stride(2)

    stride_kv_rope_b = KV_rope.stride(0)
    stride_kv_rope_h = KV_rope.stride(1)
    stride_kv_rope_d = KV_rope.stride(2)

    stride_o_nope_b = O_nope.stride(0)
    stride_o_nope_h = O_nope.stride(1)
    stride_o_nope_d = O_nope.stride(2)

    stride_o_rope_b = O_rope.stride(0)
    stride_o_rope_h = O_rope.stride(1)
    stride_o_rope_d = O_rope.stride(2)

    stride_destloc_b = DestLoc.stride(0)
    stride_destloc_s = DestLoc.stride(1)

    grid = (batch_size * head_num * tl.cdiv(seq_len, 128) * tl.cdiv(dim, 128),)

    _fwd_kernel_destindex_copy_kv[grid](
        KV_nope, KV_rope, O_nope, O_rope, DestLoc,
        batch_size, seq_len, head_num, dim,
        stride_kv_nope_b, stride_kv_nope_h, stride_kv_nope_d,
        stride_kv_rope_b, stride_kv_rope_h, stride_kv_rope_d,
        stride_o_nope_b, stride_o_nope_h, stride_o_nope_d,
        stride_o_rope_b, stride_o_rope_h, stride_o_rope_d,
        stride_destloc_b, stride_destloc_s,
        BLOCK_SIZE_SEQ=128, BLOCK_SIZE_DIM=128
    )
