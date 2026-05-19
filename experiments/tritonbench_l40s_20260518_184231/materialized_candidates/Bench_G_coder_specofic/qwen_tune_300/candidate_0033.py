import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_destindex_copy_quantize_kv(
    K, DestLoc, Out, Out_scale,
    stride_k_bs, stride_k_h, stride_k_d,
    stride_o_bs, stride_o_h, stride_o_d,
    stride_os_bs, stride_os_h, stride_os_d,
    head_num,
    cur_seq_len,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_HEAD: tl.constexpr,
):
    cur_seq_len = cur_seq_len
    cur_batch = tl.program_id(0)
    head_start = tl.program_id(1)
    head_slice = tl.arange(0, BLOCK_HEAD)
    head_mask = head_slice < head_num
    head_offset = head_start * BLOCK_HEAD + head_slice
    d_offset = tl.arange(0, BLOCK_DMODEL)

    dest_index = tl.load(DestLoc + cur_batch * cur_seq_len)

    kv_ptr = K + cur_batch * stride_k_bs + head_offset[:, None] * stride_k_h + d_offset[None, :] * stride_k_d
    o_ptr = Out + cur_batch * stride_o_bs + head_offset[:, None] * stride_o_h + d_offset[None, :] * stride_o_d
    os_ptr = Out_scale + cur_batch * stride_os_bs + head_offset[:, None] * stride_os_h + d_offset[None, :] * stride_os_d

    kv = tl.load(kv_ptr, mask=head_mask[:, None], other=0.0)
    abs_max = tl.max(tl.abs(kv), axis=1)
    scale = tl.where(abs_max <= 127, 127.0 / abs_max, 1.0)

    o = tl.dot(kv.to(tl.float32) * scale[:, None], tl.CONSTANT(1.0 / 1.0))
    os = scale

    tl.store(o_ptr + dest_index * stride_o_bs, o, mask=head_mask[:, None])
    tl.store(os_ptr + dest_index * stride_os_bs, os, mask=head_mask[:, None])
    return


@torch.no_grad()
def destindex_copy_quantize_kv(K, DestLoc, Out, Out_scale):
    seq_len = DestLoc.shape[1]
    head_num = K.shape[1]
    head_dim = K.shape[2]
    assert K.shape[1] == Out.shape[1] and K.shape[2] == Out.shape[2]
    assert K.shape[1] == Out_scale.shape[1] and K.shape[2] == Out_scale.shape[2]

    grid = (DestLoc.shape[0], triton.cdiv(head_num, 1), 1)
    num_warps = 1

    _fwd_kernel_destindex_copy_quantize_kv[grid](
        K, DestLoc, Out, Out_scale,
        K.stride(0), K.stride(1), K.stride(2),
        Out.stride(0), Out.stride(1), Out.stride(2),
        Out_scale.stride(0), Out_scale.stride(1), Out_scale.stride(2),
        head_num,
        seq_len,
        BLOCK_DMODEL=head_dim,
        BLOCK_HEAD=1,
        num_warps=num_warps,
        num_stages=1,
    )
    return
