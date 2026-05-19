import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_destindex_copy_quantize_kv(
    K, Dest_loc, Out, Out_scale,
    stride_k_bs, stride_k_h, stride_k_d,
    stride_dest_bs, stride_dest_h, stride_dest_d,
    stride_out_bs, stride_out_h, stride_out_d,
    seq_len,
    BLOCK_HEAD: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    cur_index = tl.program_id(0)
    offs_head = tl.arange(0, BLOCK_HEAD)
    offs_dmodel = tl.arange(0, BLOCK_DMODEL)

    dest_index = tl.load(Dest_loc + cur_index * stride_dest_d + offs_head[:, None] * stride_dest_h + offs_dmodel[None, :])
    k_value = tl.load(K + cur_index * stride_k_d + offs_head[:, None] * stride_k_h + offs_dmodel[None, :])

    abs_k = tl.abs(k_value)
    max_abs_k = tl.max(tl.max(abs_k, 1), 0)

    scale = max_abs_k / 127.0

    quantize_k = tl.math.round(k_value / scale)
    quantize_k = quantize_k.to(tl.int8)

    out_scale = dest_index * 2 + seq_len
    tl.store(Out + out_scale * stride_out_d + offs_head[:, None] * stride_out_h + offs_dmodel[None, :], quantize_k)
    tl.store(Out_scale + out_scale * stride_out_d + offs_head[:, None] * stride_out_h + offs_dmodel[None, :], scale)


@torch.inference_mode()
def destindex_copy_quantize_kv(K, DestLoc):
    """
    Args:
        K (torch.Tensor): shape is [batch, seq_len, num_heads, head_dim]
        DestLoc (torch.Tensor): shape is [batch, seq_len, num_heads, head_dim]
    Return:
        torch.Tensor, torch.Tensor:
            Out: shape is [batch, seq_len, num_heads, head_dim]
            Out_scale: shape is [batch, seq_len, num_heads, head_dim]
    """
    BLOCK_HEAD = 1
    BLOCK_DMODEL = 256

    batch, seq_len, num_heads, head_dim = K.shape

    Out = K.new_empty(batch, seq_len, num_heads, head_dim)
    Out_scale = K.new_empty(batch, seq_len, num_heads, head_dim)

    grid = (K.numel() // (head_dim * num_heads),)

    _fwd_kernel_destindex_copy_quantize_kv[grid](
        K, DestLoc,
        Out, Out_scale,
        K.stride(0), K.stride(1), K.stride(2),
        DestLoc.stride(0), DestLoc.stride(1), DestLoc.stride(2),
        Out.stride(0), Out.stride(1), Out.stride(2),
        seq_len,
        BLOCK_HEAD=BLOCK_HEAD,
        BLOCK_DMODEL=BLOCK_DMODEL,
        num_warps=8,
        num_stages=2,
    )

    return Out, Out_scale
