import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_destindex_copy_kv(
    KV_nope, KV_rope, DestLoc, O_nope, O_rope,
    stride_kv_bs, stride_kv_h, stride_kv_d,
    stride_o_bs, stride_o_h, stride_o_d,
    head_num, head_dim,
    BLOCK_DMODEL_NOPE: tl.constexpr,
    BLOCK_DMODEL_ROPE: tl.constexpr,
    BLOCK_HEAD: tl.constexpr,
):
    cur_index = tl.program_id(0)
    offs_h = tl.arange(0, BLOCK_HEAD)
    offs_d_nope = tl.arange(0, BLOCK_DMODEL_NOPE)
    offs_d_rope = tl.arange(0, BLOCK_DMODEL_ROPE)

    dest_index = tl.load(DestLoc + cur_index)

    # Load and store KV_nope
    kv_nope_ptrs = KV_nope + cur_index * stride_kv_bs + stride_kv_h * offs_h[:, None] + stride_kv_d * offs_d_nope[None, :]
    o_nope_ptrs = O_nope + dest_index * stride_o_bs + stride_o_h * offs_h[:, None] + stride_o_d * offs_d_nope[None, :]
    kv_nope = tl.load(kv_nope_ptrs, mask=(offs_h[:, None] < head_num) & (offs_d_nope[None, :] < head_dim), other=0.0)
    tl.store(o_nope_ptrs, kv_nope, mask=(offs_h[:, None] < head_num) & (offs_d_nope[None, :] < head_dim))

    # Load and store KV_rope
    kv_rope_ptrs = KV_rope + cur_index * stride_kv_bs + stride_kv_h * offs_h[:, None] + stride_kv_d * offs_d_rope[None, :]
    o_rope_ptrs = O_rope + dest_index * stride_o_bs + stride_o_h * offs_h[:, None] + stride_o_d * offs_d_rope[None, :]
    kv_rope = tl.load(kv_rope_ptrs, mask=(offs_h[:, None] < head_num) & (offs_d_rope[None, :] < head_dim), other=0.0)
    tl.store(o_rope_ptrs, kv_rope, mask=(offs_h[:, None] < head_num) & (offs_d_rope[None, :] < head_dim))

@torch.no_grad()
def destindex_copy_kv(KV_nope, KV_rope, DestLoc, O_nope, O_rope):
    seq_len = DestLoc.shape[0]
    head_num = KV_nope.shape[1]
    head_dim_nope = KV_nope.shape[2]
    head_dim_rope = KV_rope.shape[2]
    assert KV_nope.shape[1] == O_nope.shape[1] and KV_nope.shape[2] == O_nope.shape[2]
    assert KV_rope.shape[1] == O_rope.shape[1] and KV_rope.shape[2] == O_rope.shape[2]
    BLOCK_HEAD = triton.next_power_of_2(head_num)
    BLOCK_DMODEL_NOPE = triton.next_power_of_2(head_dim_nope)
    BLOCK_DMODEL_ROPE = triton.next_power_of_2(head_dim_rope)
    grid = (seq_len,)
    num_warps = 4

    _fwd_kernel_destindex_copy_kv[grid](
        KV_nope, KV_rope, DestLoc, O_nope, O_rope,
        KV_nope.stride(0), KV_nope.stride(1), KV_nope.stride(2),
        O_nope.stride(0), O_nope.stride(1), O_nope.stride(2),
        head_num, head_dim_nope,
        BLOCK_DMODEL_NOPE=BLOCK_DMODEL_NOPE,
        BLOCK_DMODEL_ROPE=BLOCK_DMODEL_ROPE,
        BLOCK_HEAD=BLOCK_HEAD,
        num_warps=num_warps,
        num_stages=1,
    )

@triton.jit
def _fwd_kernel_destindex_copy_quantize_kv(
    KV_nope, KV_rope, DestLoc, O_nope, O_rope, O_scale_nope, O_scale_rope,
    stride_kv_bs, stride_kv_h, stride_kv_d,
    stride_o_bs, stride_o_h, stride_o_d,
    stride_os_bs, stride_os_h,
    head_num, head_dim_nope, head_dim_rope,
    BLOCK_DMODEL_NOPE: tl.constexpr,
    BLOCK_DMODEL_ROPE: tl.constexpr,
    BLOCK_HEAD: tl.constexpr,
):
    cur_index = tl.program_id(0)
    offs_h = tl.arange(0, BLOCK_HEAD)
    offs_d_nope = tl.arange(0, BLOCK_DMODEL_NOPE)
    offs_d_rope = tl.arange(0, BLOCK_DMODEL_ROPE)

    dest_index = tl.load(DestLoc + cur_index)

    # Process KV_nope
    src_data_nope = tl.load(
        KV_nope + cur_index * stride_kv_bs + offs_h[:, None] * stride_kv_h + stride_kv_d * offs_d_nope[None, :],
        mask=(offs_h[:, None] < head_num) & (offs_d_nope[None, :] < head_dim_nope),
        other=0.0,
    )
    abs_data_nope = tl.abs(src_data_nope)
    data_scale_nope = (tl.max(abs_data_nope, axis=1) / 127.0).to(O_scale_nope.dtype.element_ty)[:, None]
    q_src_data_nope = (src_data_nope / data_scale_nope).to(tl.int8)
    o_ptrs_nope = O_nope + dest_index * stride_o_bs + stride_o_h * offs_h[:, None] + stride_o_d * offs_d_nope[None, :]
    os_ptrs_nope = O_scale_nope + dest_index * stride_os_bs + stride_os_h * offs_h[:, None]
    tl.store(o_ptrs_nope, q_src_data_nope, mask=(offs_h[:, None] < head_num) & (offs_d_nope[None, :] < head_dim_nope))
    tl.store(os_ptrs_nope, data_scale_nope, mask=(offs_h[:, None] < head_num))

    # Process KV_rope
    src_data_rope = tl.load(
        KV_rope + cur_index * stride_kv_bs + offs_h[:, None] * stride_kv_h + stride_kv_d * offs_d_rope[None, :],
        mask=(offs_h[:, None] < head_num) & (offs_d_rope[None, :] < head_dim_rope),
        other=0.0,
    )
    abs_data_rope = tl.abs(src_data_rope)
    data_scale_rope = (tl.max(abs_data_rope, axis=1) / 127.0).to(O_scale_rope.dtype.element_ty)[:, None]
    q_src_data_rope = (src_data_rope / data_scale_rope).to(tl.int8)
    o_ptrs_rope = O_rope + dest_index * stride_o_bs + stride_o_h * offs_h[:, None] + stride_o_d * offs_d_rope[None, :]
    os_ptrs_rope = O_scale_rope + dest_index * stride_os_bs + stride_os_h * offs_h[:, None]
    tl.store(o_ptrs_rope, q_src_data_rope, mask=(offs_h[:, None] < head_num) & (offs_d_rope[None, :] < head_dim_rope))
    tl.store(os_ptrs_rope, data_scale_rope, mask=(offs_h[:, None] < head_num))

@torch.no_grad()
def destindex_copy_quantize_kv(KV_nope, KV_rope, DestLoc, O_nope, O_rope, O_scale_nope, O_scale_rope):
    seq_len = DestLoc.shape[0]
    head_num = KV_nope.shape[1]
    head_dim_nope = KV_nope.shape[2]
    head_dim_rope = KV_rope.shape[2]
    assert KV_nope.shape[1] == O_nope.shape[1] and KV_nope.shape[2] == O_nope.shape[2]
    assert KV_rope.shape[1] == O_rope.shape[1] and KV_rope.shape[2] == O_rope.shape[2]
    BLOCK_HEAD = triton.next_power_of_2(head_num)
    BLOCK_DMODEL_NOPE = triton.next_power_of_2(head_dim_nope)
    BLOCK_DMODEL_ROPE = triton.next_power_of_2(head_dim_rope)
    grid = (seq_len,)
    num_warps = 4

    _fwd_kernel_destindex_copy_quantize_kv[grid](
        KV_nope, KV_rope, DestLoc, O_nope, O_rope, O_scale_nope, O_scale_rope,
        KV_nope.stride(0), KV_nope.stride(1), KV_nope.stride(2),
        O_nope.stride(0), O_nope.stride(1), O_nope.stride(2),
        O_scale_nope.stride(0), O_scale_nope.stride(1),
        head_num, head_dim_nope, head_dim_rope,
        BLOCK_DMODEL_NOPE=BLOCK_DMODEL_NOPE,
        BLOCK_DMODEL_ROPE=BLOCK_DMODEL_ROPE,
        BLOCK_HEAD=BLOCK_HEAD,
        num_warps=num_warps,
        num_stages=1,
    )
