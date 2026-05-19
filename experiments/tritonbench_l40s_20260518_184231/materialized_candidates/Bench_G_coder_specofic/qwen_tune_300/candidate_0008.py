import torch
import triton
import triton.language as tl


@triton.jit
def _fwd_kernel_destindex_copy_quantize_kv(
    K, Dest_loc, Out, Out_scale,
    stride_k_bs, stride_k_h, stride_k_d,
    stride_o_bs, stride_o_h, stride_o_d,
    stride_os_bs, stride_os_h, stride_os_g, stride_os_d,
    stride_o_scale_bs, stride_o_scale_h, stride_o_scale_g,
    gqa_group_size,
    seq_len,
    cache_key_group_num: tl.constexpr,
    BLOCK_GROUP_NUM: tl.constexpr,
    BLOCK_GROUP_DIM: tl.constexpr,
):
    cur_index = tl.program_id(0)
    cur_head = tl.program_id(1)

    cur_head_striding = cur_head * stride_k_h
    cur_batch_striding = cur_index * stride_k_bs

    src_data_offset = cur_batch_striding + cur_head_striding
    dest_index = tl.load(Dest_loc + cur_batch_striding + cur_head_striding + tl.arange(0, BLOCK_GROUP_NUM)[:, None] * stride_k_h + tl.arange(0, 16)[None, :])

    # Load source data
    k_data = tl.load(K + src_data_offset + tl.arange(0, BLOCK_GROUP_NUM)[:, None] * stride_k_h + tl.arange(0, 16)[None, :] * stride_k_d)

    # Compute absmax
    absmax = tl.zeros([BLOCK_GROUP_NUM, 16], dtype=tl.float32)
    for _ in range(0, tl.cdiv(seq_len, 16)):
        src_data_ptr = K + src_data_offset + tl.arange(0, BLOCK_GROUP_NUM)[:, None] * stride_k_h + (_ * 16 + tl.arange(0, 16))[None, :] * stride_k_d
        src_data = tl.load(src_data_ptr)
        absmax = tl.maximum(tl.abs(src_data), absmax)
    absmax = tl.max(tl.reshape(absmax, [BLOCK_GROUP_NUM * 16]), axis=0)

    # Quantize
    scale = (absmax / 127.).to(tl.float16)
    int_k = (k_data / scale[:, None]).to(tl.int8)

    # Store
    out_offset = cur_batch_striding + cur_head_striding + tl.arange(0, BLOCK_GROUP_NUM)[:, None] * stride_os_g + tl.arange(0, 16)[None, :] 
    tl.store(Out + out_offset, int_k, mask=dest_index[:, None] < 0)
    tl.store(Out_scale + cur_batch_striding + cur_head_striding + tl.arange(0, BLOCK_GROUP_NUM)[:, None] * stride_o_scale_g, scale[:, None])
    return


@torch.no_grad()
def destindex_copy_quantize_kv(K, DestLoc, Out, Out_scale, group_size):
    seq_len = K.shape[2]
    head_num = K.shape[1]
    head_dim = K.shape[3]
    assert head_dim % group_size == 0, "error head dim, can not been supported to copy quantize"

    # using constexpr to avoid generating multiple kernel
    BLOCK_GROUP_NUM = 16
    BLOCK_GROUP_DIM = triton.next_power_of_2(group_size)

    # split head by group size
    head_group_num = head_dim // group_size
    K = K.view((K.shape[0], K.shape[1], head_group_num, group_size))
    dest_index = DestLoc // group_size
    dest_index_flattened = dest_index.view(-1)

    grid = (triton.cdiv(DestLoc.shape[1] * DestLoc.shape[2], BLOCK_GROUP_NUM), K.shape[1])
    _fwd_kernel_destindex_copy_quantize_kv[grid](
        K, DestLoc, Out, Out_scale,
        K.stride(0), K.stride(1), K.stride(2),
        Out.stride(0), Out.stride(1), Out.stride(2),
        Out.stride(0), Out.stride(1), Out.stride(3), Out.stride(4),
        Out_scale.stride(0), Out_scale.stride(1), Out_scale.stride(2),
        group_size,
        seq_len,
        16,
        BLOCK_GROUP_NUM,
        BLOCK_GROUP_DIM,
    )
    return
