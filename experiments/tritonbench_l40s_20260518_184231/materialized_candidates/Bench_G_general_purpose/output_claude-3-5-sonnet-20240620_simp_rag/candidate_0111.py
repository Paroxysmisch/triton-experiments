import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_destindex_copy_kv(
    KV_nope, KV_rope, DestLoc, O_nope, O_rope,
    stride_kv_bs, stride_kv_h, stride_kv_d,
    stride_o_bs, stride_o_h, stride_o_d,
    head_num, head_dim,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_HEAD: tl.constexpr,
):
    cur_index = tl.program_id(0)
    offs_h = tl.arange(0, BLOCK_HEAD)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    dest_index = tl.load(DestLoc + cur_index)

    # Load from KV_nope and store to O_nope
    kv_nope_ptrs = KV_nope + cur_index * stride_kv_bs + stride_kv_h * offs_h[:, None] + stride_kv_d * offs_d[None, :]
    o_nope_ptrs = O_nope + dest_index * stride_o_bs + stride_o_h * offs_h[:, None] + stride_o_d * offs_d[None, :]

    kv_nope = tl.load(kv_nope_ptrs, mask=(offs_h[:, None] < head_num) & (offs_d[None, :] < head_dim), other=0.0)
    tl.store(o_nope_ptrs, kv_nope, mask=(offs_h[:, None] < head_num) & (offs_d[None, :] < head_dim))

    # Load from KV_rope and store to O_rope
    kv_rope_ptrs = KV_rope + cur_index * stride_kv_bs + stride_kv_h * offs_h[:, None] + stride_kv_d * offs_d[None, :]
    o_rope_ptrs = O_rope + dest_index * stride_o_bs + stride_o_h * offs_h[:, None] + stride_o_d * offs_d[None, :]

    kv_rope = tl.load(kv_rope_ptrs, mask=(offs_h[:, None] < head_num) & (offs_d[None, :] < head_dim), other=0.0)
    tl.store(o_rope_ptrs, kv_rope, mask=(offs_h[:, None] < head_num) & (offs_d[None, :] < head_dim))

@torch.no_grad()
def destindex_copy_kv(KV_nope, KV_rope, DestLoc, O_nope, O_rope):
    seq_len = DestLoc.shape[0]
    head_num = KV_nope.shape[1]
    head_dim = KV_nope.shape[2]
    assert KV_nope.shape == KV_rope.shape == O_nope.shape == O_rope.shape
    BLOCK_HEAD = triton.next_power_of_2(head_num)
    BLOCK_DMODEL = triton.next_power_of_2(head_dim)
    grid = (seq_len,)
    num_warps = 4

    _fwd_kernel_destindex_copy_kv[grid](
        KV_nope, KV_rope, DestLoc, O_nope, O_rope,
        KV_nope.stride(0), KV_nope.stride(1), KV_nope.stride(2),
        O_nope.stride(0), O_nope.stride(1), O_nope.stride(2),
        head_num, head_dim,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_HEAD=BLOCK_HEAD,
        num_warps=num_warps,
        num_stages=1,
    )
    return
