import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_destindex_copy_kv(
    K, Out, Dest_loc,
    stride_k_b, stride_k_h, stride_k_d,
    stride_o_b, stride_o_h, stride_o_d,
    stride_d_b, stride_d_s,
    batch_size, seq_len, head_num, depth,
    BLOCK_HEAD: tl.constexpr
):
    pid = tl.program_id(0)
    if pid >= seq_len:
        return

    dest_loc = tl.load(Dest_loc + pid * stride_d_b + tl.arange(0, BLOCK_HEAD) * stride_d_s)
    dest_loc = tl.max(dest_loc, 0)

    offs_h = tl.arange(0, BLOCK_HEAD)
    offs_d = tl.arange(0, depth)

    k_ptrs = K + (pid * stride_k_b)[:, None, None] + (offs_h[None, :, None] * stride_k_h) + (offs_d[None, None, :] * stride_k_d)
    o_ptrs = Out + (pid * stride_o_b)[:, None, None] + (dest_loc[:, :, None] * stride_o_h) + (offs_d[None, None, :] * stride_o_d)

    k_vals = tl.load(k_ptrs, mask=offs_h[None, :, None] < head_num)
    tl.store(o_ptrs, k_vals, mask=offs_h[None, :, None] < head_num)

import torch
import triton
import triton.language as tl

def destindex_copy_kv(K, Out, Dest_loc):
    batch_size, head_num, seq_len, depth = K.shape
    assert K.shape == Out.shape, "K and Out must have the same shape"
    assert Dest_loc.shape == (batch_size, seq_len), "Dest_loc must have shape (batch_size, seq_len)"

    stride_k_b = K.stride(0)
    stride_k_h = K.stride(1)
    stride_k_d = K.stride(3)

    stride_o_b = Out.stride(0)
    stride_o_h = Out.stride(1)
    stride_o_d = Out.stride(3)

    stride_d_b = Dest_loc.stride(0)
    stride_d_s = Dest_loc.stride(1)

    BLOCK_HEAD = 1 << (head_num - 1).bit_length()  # Next power of 2

    grid = (seq_len,)

    _fwd_kernel_destindex_copy_kv[grid](
        K, Out, Dest_loc,
        stride_k_b, stride_k_h, stride_k_d,
        stride_o_b, stride_o_h, stride_o_d,
        stride_d_b, stride_d_s,
        batch_size, seq_len, head_num, depth,
        BLOCK_HEAD
    )
