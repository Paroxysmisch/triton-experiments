import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_destindex_copy_kv(
    KV_nope,
    KV_rope,
    Dest_loc,
    O_nope,
    O_rope,
    stride_kn_bs,
    stride_kn_h,
    stride_kn_d,
    stride_kr_bs,
    stride_kr_h,
    stride_kr_d,
    stride_on_bs,
    stride_on_h,
    stride_on_d,
    stride_or_bs,
    stride_or_h,
    stride_or_d,
    head_num_nope,
    head_dim_nope,
    head_num_rope,
    head_dim_rope,
    BLOCK_DMODEL_NOPE: tl.constexpr,
    BLOCK_DMODEL_ROPE: tl.constexpr,
    BLOCK_HEAD: tl.constexpr,
):
    cur_index = tl.program_id(0)
    offs_h = tl.arange(0, BLOCK_HEAD)
    offs_d_nope = tl.arange(0, BLOCK_DMODEL_NOPE)
    offs_d_rope = tl.arange(0, BLOCK_DMODEL_ROPE)

    dest_index = tl.load(Dest_loc + cur_index)

    # Process KV_nope and O_nope
    kn_ptrs = (
        KV_nope 
        + cur_index * stride_kn_bs 
        + offs_h[:, None] * stride_kn_h 
        + offs_d_nope[None, :] * stride_kn_d
    )
    on_ptrs = (
        O_nope 
        + dest_index * stride_on_bs 
        + offs_h[:, None] * stride_on_h 
        + offs_d_nope[None, :] * stride_on_d
    )
    mask_nope = (offs_h[:, None] < head_num_nope) & (offs_d_nope[None, :] < head_dim_nope)
    k_nope = tl.load(kn_ptrs, mask=mask_nope, other=0.0)
    tl.store(on_ptrs, k_nope, mask=mask_nope)

    # Process KV_rope and O_rope
    kr_ptrs = (
        KV_rope 
        + cur_index * stride_kr_bs 
        + offs_h[:, None] * stride_kr_h 
        + offs_d_rope[None, :] * stride_kr_d
    )
    or_ptrs = (
        O_rope 
        + dest_index * stride_or_bs 
        + offs_h[:, None] * stride_or_h 
        + offs_d_rope[None, :] * stride_or_d
    )
    mask_rope = (offs_h[:, None] < head_num_rope) & (offs_d_rope[None, :] < head_dim_rope)
    k_rope = tl.load(kr_ptrs, mask=mask_rope, other=0.0)
    tl.store(or_ptrs, k_rope, mask=mask_rope)

@torch.no_grad()
def destindex_copy_kv(KV_nope, KV_rope, DestLoc, O_nope, O_rope):
    seq_len = DestLoc.shape[0]
    # Ensure source and destination shapes match
    assert KV_nope.shape == O_nope.shape, "KV_nope and O_nope must have the same shape"
    assert KV_rope.shape == O_rope.shape, "KV_rope and O_rope must have the same shape"
    
    head_num_nope, head_dim_nope = KV_nope.shape[1], KV_nope.shape[2]
    head_num_rope, head_dim_rope = KV_rope.shape[1], KV_rope.shape[2]
    
    # Determine block sizes
    BLOCK_HEAD = triton.next_power_of_2(max(head_num_nope, head_num_rope))
    BLOCK_DMODEL_NOPE = triton.next_power_of_2(head_dim_nope)
    BLOCK_DMODEL_ROPE = triton.next_power_of_2(head_dim_rope)
    
    grid = (seq_len,)
    num_warps = 1
    
    # Launch kernel with appropriate parameters
    _fwd_kernel_destindex_copy_kv[grid](
        KV_nope, KV_rope, DestLoc, O_nope, O_rope,
        KV_nope.stride(0), KV_nope.stride(1), KV_nope.stride(2),
        KV_rope.stride(0), KV_rope.stride(1), KV_rope.stride(2),
        O_nope.stride(0), O_nope.stride(1), O_nope.stride(2),
        O_rope.stride(0), O_rope.stride(1), O_rope.stride(2),
        head_num_nope, head_dim_nope,
        head_num_rope, head_dim_rope,
        BLOCK_DMODEL_NOPE=BLOCK_DMODEL_NOPE,
        BLOCK_DMODEL_ROPE=BLOCK_DMODEL_ROPE,
        BLOCK_HEAD=BLOCK_HEAD,
        num_warps=num_warps,
        num_stages=1
    )
    return
