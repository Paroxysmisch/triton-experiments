import torch
import triton
import triton.language as tl
from torch import Tensor

@triton.jit
def _fwd_kernel_destindex_copy_kv(
    KV_nope,
    KV_rope,
    DestLoc,
    O_nope,
    O_rope,
    BLOCK_DMODEL_NOPE: tl.constexpr,
    BLOCK_DMODEL_ROPE: tl.constexpr,
):
    seq_index = tl.program_id(0)
    offs_nope = tl.arange(0, BLOCK_DMODEL_NOPE)
    offs_rope = tl.arange(0, BLOCK_DMODEL_ROPE)
    dest_index = tl.load(DestLoc + seq_index)
    kv_nope_ptrs = KV_nope + seq_index * BLOCK_DMODEL_NOPE + offs_nope
    kv_rope_ptrs = KV_rope + seq_index * BLOCK_DMODEL_ROPE + offs_rope
    o_nope_ptrs = O_nope + dest_index * BLOCK_DMODEL_NOPE + offs_nope
    o_rope_ptrs = O_rope + dest_index * BLOCK_DMODEL_ROPE + offs_rope
    kv_nope = tl.load(kv_nope_ptrs, mask=offs_nope < BLOCK_DMODEL_NOPE, other=0.0)
    kv_rope = tl.load(kv_rope_ptrs, mask=offs_rope < BLOCK_DMODEL_ROPE, other=0.0)
    tl.store(o_nope_ptrs, kv_nope, mask=offs_nope < BLOCK_DMODEL_NOPE)
    tl.store(o_rope_ptrs, kv_rope, mask=offs_rope < BLOCK_DMODEL_ROPE)
    return

@torch.inference_mode()
def destindex_copy_kv(KV_nope: Tensor, KV_rope: Tensor, DestLoc: Tensor, O_nope: Tensor, O_rope: Tensor):
    """
    KV_nope: (batch_size, num_heads, head_dim)
    KV_rope: (batch_size, num_heads, head_dim)
    DestLoc: (batch_size, num_heads)
    O_nope: (batch_size, num_heads, head_dim)
    O_rope: (batch_size, num_heads, head_dim)
    """
    assert KV_nope.shape == O_nope.shape
    assert KV_rope.shape == O_rope.shape
    assert KV_nope.shape[:-1] == DestLoc.shape
    assert KV_nope.is_contiguous()
    assert KV_rope.is_contiguous()
    assert O_nope.is_contiguous()
    assert O_rope.is_contiguous()
    batch_size, num_heads, head_dim = KV_nope.shape
    num_warps = 4
    if head_dim >= 2048:
        num_warps = 8
    if head_dim >= 4096:
        num_warps = 16
    BLOCK_DMODEL_NOPE = triton.next_power_of_2(head_dim)
    BLOCK_DMODEL_ROPE = triton.next_power_of_2(head_dim)
    _fwd_kernel_destindex_copy_kv[(batch_size * num_heads,)](
        KV_nope,
        KV_rope,
        DestLoc,
        O_nope,
        O_rope,
        BLOCK_DMODEL_NOPE=BLOCK_DMODEL_NOPE,
        BLOCK_DMODEL_ROPE=BLOCK_DMODEL_ROPE,
        num_warps=num_warps,
        num_stages=1,
    )
    return O_nope, O_rope
