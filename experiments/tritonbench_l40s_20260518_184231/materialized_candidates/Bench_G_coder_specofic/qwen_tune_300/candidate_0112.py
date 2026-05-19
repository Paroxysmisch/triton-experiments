import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_destindex_copy_kv(
    KV_nope,
    KV_rope,
    DestLoc,
    O_nope,
    O_rope,
    stride_kv_nope_bs,
    stride_kv_nope_h,
    stride_kv_nope_d,
    stride_kv_rope_bs,
    stride_kv_rope_h,
    stride_kv_rope_d,
    stride_o_nope_bs,
    stride_o_nope_h,
    stride_o_nope_d,
    stride_o_rope_bs,
    stride_o_rope_h,
    stride_o_rope_d,
    cur_index,
    BLOCK_DMODEL_NOPE: tl.constexpr,
    BLOCK_DMODEL_ROPE: tl.constexpr,
):
    cur_index = cur_index.to(tl.int64)

    offs_h = tl.arange(0, BLOCK_DMODEL_NOPE)
    offs_d = tl.arange(0, BLOCK_DMODEL_ROPE)

    kv_nope_ptrs = (
        KV_nope
        + cur_index * stride_kv_nope_bs
        + stride_kv_nope_h * offs_h[:, None]
        + stride_kv_nope_d * offs_d[None, :]
    )
    kv_rope_ptrs = (
        KV_rope
        + cur_index * stride_kv_rope_bs
        + stride_kv_rope_h * offs_h[:, None]
        + stride_kv_rope_d * offs_d[None, :]
    )

    o_nope_ptrs = (
        O_nope
        + cur_index * stride_o_nope_bs
        + stride_o_nope_h * offs_h[:, None]
        + stride_o_nope_d * offs_d[None, :]
    )
    o_rope_ptrs = (
        O_rope
        + cur_index * stride_o_rope_bs
        + stride_o_rope_h * offs_h[:, None]
        + stride_o_rope_d * offs_d[None, :]
    )

    kv_nope = tl.load(kv_nope_ptrs, mask=offs_h[:, None] < BLOCK_DMODEL_NOPE, other=0.0)
    kv_rope = tl.load(kv_rope_ptrs, mask=offs_h[:, None] < BLOCK_DMODEL_ROPE, other=0.0)

    dest_index = tl.load(DestLoc + cur_index)
    o_nope_ptrs = (
        O_nope
        + dest_index * stride_o_nope_bs
        + stride_o_nope_h * offs_h[:, None]
        + stride_o_nope_d * offs_d[None, :]
    )
    o_rope_ptrs = (
        O_rope
        + dest_index * stride_o_rope_bs
        + stride_o_rope_h * offs_h[:, None]
        + stride_o_rope_d * offs_d[None, :]
    )

    tl.store(o_nope_ptrs, kv_nope, mask=offs_h[:, None] < BLOCK_DMODEL_NOPE)
    tl.store(o_rope_ptrs, kv_rope, mask=offs_h[:, None] < BLOCK_DMODEL_ROPE)

    return

@torch.no_grad()
def destindex_copy_kv(KV_nope, KV_rope, DestLoc, O_nope, O_rope):
    seq_len = DestLoc.shape[0]
    assert KV_nope.shape[0] == seq_len
    assert KV_rope.shape[0] == seq_len
    assert DestLoc.shape[0] == O_nope.shape[0]
    assert DestLoc.shape[0] == O_rope.shape[0]

    batch, head, dim = KV_nope.shape
    assert dim == O_nope.shape[2]
    assert KV_rope.shape[2] == O_rope.shape[2]

    aligned_dim = triton.next_power_of_2(dim)
    grid = (seq_len,)
    num_warps = 1

    _fwd_kernel_destindex_copy_kv[grid](
        KV_nope,
        KV_rope,
        DestLoc,
        O_nope,
        O_rope,
        KV_nope.stride(0),
        KV_nope.stride(1),
        KV_nope.stride(2),
        KV_rope.stride(0),
        KV_rope.stride(1),
        KV_rope.stride(2),
        O_nope.stride(0),
        O_nope.stride(1),
        O_nope.stride(2),
        O_rope.stride(0),
        O_rope.stride(1),
        O_rope.stride(2),
        BLOCK_DMODEL_NOPE=aligned_dim,
        BLOCK_DMODEL_ROPE=aligned_dim,
        num_warps=num_warps,
        num_stages=1,
    )
    return
