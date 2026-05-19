import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_destindex_copy_quantize_kv(
    K,
    Out,
    Out_scale,
    DestLoc,
    head_num,
    seq_len,
    stride_k_bs,
    stride_k_h,
    stride_k_d,
    stride_o_bs,
    stride_o_h,
    stride_o_d,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_HEAD: tl.constexpr,
):
    cur_head_index = tl.program_id(0)
    cur_sm_index = tl.program_id(1)
    cur_kv_head_index = cur_head_index // head_num
    grid_m = tl.cdiv(seq_len, BLOCK_DMODEL)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_head = tl.arange(0, BLOCK_HEAD) + cur_head_index * BLOCK_HEAD
    offs_sm = tl.arange(0, 1)
    offs = offs_d + (offs_head[:, None] * BLOCK_DMODEL + offs_sm[None, :] * BLOCK_DMODEL * head_num)

    k_ptrs = K + offs + cur_kv_head_index * stride_k_h
    dest_index_ptrs = DestLoc + cur_sm_index * seq_len + tl.arange(0, BLOCK_DMODEL)
    o_ptrs = Out + offs + cur_kv_head_index * stride_o_h
    o_scale_ptrs = Out_scale + cur_head_index * head_num + cur_kv_head_index

    k = tl.load(k_ptrs, mask=(offs_d[None, :] < seq_len), other=0.0)
    dest_index = tl.load(dest_index_ptrs, mask=(offs_d[None, :] < seq_len), other=0)

    abs_k = tl.abs(k)
    abs_max = tl.max(abs_k, axis=0)
    scale = (abs_max / 127.0).to(o_scale_ptrs.dtype.element_ty)
    q_k = (k / scale).to(tl.int8)
    tl.store(o_ptrs, q_k, mask=(offs_d[None, :] < seq_len))
    tl.store(o_scale_ptrs, scale)

def destindex_copy_quantize_kv(K, DestLoc, Out, Out_scale):
    """
    Args:
        K: (total_head_num, head_dim)
        DestLoc: (1, seq_len)
        Out: (total_head_num, head_dim)
        Out_scale: (total_head_num, head_dim)
    """
    head_num = K.shape[0]
    head_dim = K.shape[1]
    seq_len = DestLoc.shape[-1]

    BLOCK_HEAD = triton.next_power_of_2(head_dim)
    BLOCK_DMODEL = triton.next_power_of_2(head_dim)
    num_warps = 1
    assert BLOCK_DMODEL <= 2048

    grid = (head_num, seq_len // BLOCK_DMODEL)

    _fwd_kernel_destindex_copy_quantize_kv[grid](
        K,
        Out,
        Out_scale,
        DestLoc,
        head_num,
        seq_len,
        K.stride(0),
        K.stride(1),
        K.stride(2),
        Out.stride(0),
        Out.stride(1),
        Out.stride(2),
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_HEAD=BLOCK_HEAD,
        num_warps=num_warps,
        num_stages=1,
    )
