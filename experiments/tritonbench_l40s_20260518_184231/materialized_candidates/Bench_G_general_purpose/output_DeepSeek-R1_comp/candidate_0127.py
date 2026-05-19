import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_destindex_copy_quantize_kv(
    K_ptr, Dest_loc_ptr, Out_ptr, Out_scale_ptr,
    stride_k_h, stride_k_d,
    stride_out_h, stride_out_d,
    stride_scale_d, stride_scale_h,
    num_heads, d_model,
    BLOCK_HEAD: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    pid_dest = tl.program_id(0)
    pid_head_block = tl.program_id(1)
    pid_dmodel_block = tl.program_id(2)

    dest_idx = tl.load(Dest_loc_ptr + pid_dest)

    head_idx = pid_head_block * BLOCK_HEAD + tl.arange(0, BLOCK_HEAD)
    dmodel_idx = pid_dmodel_block * BLOCK_DMODEL + tl.arange(0, BLOCK_DMODEL)

    mask_head = head_idx < num_heads
    mask_dmodel = dmodel_idx < d_model
    mask = mask_head[:, None] & mask_dmodel[None, :]

    k_ptrs = K_ptr + head_idx[:, None] * stride_k_h + dmodel_idx[None, :] * stride_k_d
    k_vals = tl.load(k_ptrs, mask=mask, other=0.0)

    abs_k = tl.abs(k_vals)
    max_val = tl.max(abs_k)
    scale = max_val / 127.0

    scaled_k = (k_vals * (127.0 / max_val)).to(tl.int8)

    out_ptrs = Out_ptr + dest_idx * (num_heads * d_model) + head_idx[:, None] * stride_out_h + dmodel_idx[None, :] * stride_out_d
    tl.store(out_ptrs, scaled_k, mask=mask)

    scale_offset = dest_idx * stride_scale_d + pid_head_block * stride_scale_h + pid_dmodel_block
    scale_ptrs = Out_scale_ptr + scale_offset
    tl.store(scale_ptrs, scale, mask=mask_head[0] & mask_dmodel[0])

@torch.no_grad()
def destindex_copy_quantize_kv(K: torch.Tensor, Dest_loc: torch.Tensor, Out: torch.Tensor, Out_scale: torch.Tensor):
    assert K.is_contiguous() and Dest_loc.is_contiguous()
    num_heads, d_model = K.shape[-2], K.shape[-1]
    seq_len = Dest_loc.shape[0]

    BLOCK_HEAD = triton.next_power_of_2(num_heads)
    BLOCK_DMODEL = triton.next_power_of_2(d_model)
    BLOCK_HEAD = max(BLOCK_HEAD, 16)
    BLOCK_DMODEL = max(BLOCK_DMODEL, 16)

    num_head_blocks = (num_heads + BLOCK_HEAD - 1) // BLOCK_HEAD
    num_dmodel_blocks = (d_model + BLOCK_DMODEL - 1) // BLOCK_DMODEL

    grid = (seq_len, num_head_blocks, num_dmodel_blocks)

    _fwd_kernel_destindex_copy_quantize_kv[grid](
        K, Dest_loc, Out, Out_scale,
        K.stride(-2), K.stride(-1),
        Out.stride(-2), Out.stride(-1),
        Out_scale.stride(0), Out_scale.stride(1),
        num_heads, d_model,
        BLOCK_HEAD=BLOCK_HEAD, BLOCK_DMODEL=BLOCK_DMODEL,
        num_warps=4,
        num_stages=3,
    )
