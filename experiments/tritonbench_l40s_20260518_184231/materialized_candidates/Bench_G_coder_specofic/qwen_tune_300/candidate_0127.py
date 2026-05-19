import math
import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_destindex_copy_quantize_kv(
    K,
    Dest_loc,
    Out,
    Out_scale,
    stride_k_bs,
    stride_k_h,
    stride_k_d,
    stride_o_bs,
    stride_o_h,
    stride_o_d,
    stride_os_bs,
    stride_os_h,
    stride_os_d,
    K_HEADS,
    D_MODEL_K,
    BLOCK_HEAD: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    cur_index = tl.program_id(0)
    cur_head = tl.program_id(1)

    offs_h = cur_head * BLOCK_HEAD + tl.arange(0, BLOCK_HEAD)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    mask_h = offs_h < K_HEADS
    mask_d = offs_d < D_MODEL_K

    off_k = cur_index * BLOCK_DMODEL * K_HEADS + offs_h[:, None] * stride_k_h + offs_d[None, :] * stride_k_d
    off_dest = tl.load(Dest_loc + cur_index)
    off_o = off_dest[:, None] * stride_o_bs + offs_h[None, :] * stride_o_h + offs_d[:, None] * stride_o_d
    off_os = cur_index * BLOCK_DMODEL * K_HEADS + offs_h[:, None] * stride_os_h + offs_d[None, :] * stride_os_d

    k_ptrs = K + off_k
    o_ptrs = Out + off_o
    os_ptrs = Out_scale + off_os

    max_abs = tl.zeros([BLOCK_HEAD, BLOCK_DMODEL], dtype=tl.float32)
    for _ in range(0, tl.cdiv(D_MODEL_K, BLOCK_DMODEL)):
        k = tl.load(k_ptrs, mask=mask_h[:, None] & mask_d[None, :], other=0.0)
        abs_k = tl.abs(k)
        max_abs_new = tl.maximum(max_abs, tl.max(tl.where(mask_h[:, None] & mask_d[None, :], abs_k, 0), axis=1))
        max_abs = max_abs_new
        k_ptrs += BLOCK_DMODEL
    scale = tl.math.llrint(127.0 / max_abs)

    k_ptrs = tl.make_block_ptr(K + off_k, (K_HEADS, D_MODEL_K), (stride_k_h, stride_k_d), (cur_head, 0), (BLOCK_HEAD, BLOCK_DMODEL), (1, 0))
    os_ptrs = Out_scale + off_os

    for _ in range(0, tl.cdiv(D_MODEL_K, BLOCK_DMODEL)):
        k = tl.load(k_ptrs)
        os = scale[:, None] * k
        os = tl.math.llrint(os)
        tl.store(o_ptrs, os, mask=mask_h[:, None] & mask_d[None, :])
        tl.store(os_ptrs, scale, mask=mask_h[:, None] & mask_d[None, :])

        k_ptrs = tl.advance(k_ptrs, (0, BLOCK_DMODEL))
        o_ptrs = tl.advance(o_ptrs, (BLOCK_HEAD, 0))
        os_ptrs = tl.advance(os_ptrs, (0, BLOCK_DMODEL))

    return

@torch.no_grad()
def destindex_copy_quantize_kv(K, DestLoc, Out, Out_scale):
    seq_len = DestLoc.shape[0]
    assert K.shape[1] == Out.shape[1] and K.shape[2] == Out.shape[2]
    K_HEADS = K.shape[1]
    D_MODEL_K = K.shape[2]
    BLOCK_HEAD = triton.next_power_of_2(K_HEADS)
    BLOCK_DMODEL = triton.next_power_of_2(D_MODEL_K)
    assert D_MODEL_K in {16, 32, 64, 128, 256, 512}
    if BLOCK_HEAD >= 64:
        BLOCK_HEAD = 64
    if BLOCK_HEAD >= 32:
        BLOCK_DMODEL = 32
    else:
        BLOCK_DMODEL = 64

    grid = (seq_len, triton.cdiv(K_HEADS, BLOCK_HEAD))
    num_warps = 1
    num_stages = 4 if D_MODEL_K < 64 else 3

    _fwd_kernel_destindex_copy_quantize_kv[grid](
        K,
        DestLoc,
        Out,
        Out_scale,
        K.stride(0),
        K.stride(1),
        K.stride(2),
        Out.stride(0),
        Out.stride(1),
        Out.stride(2),
        Out_scale.stride(0),
        Out_scale.stride(1),
        Out_scale.stride(2),
        K_HEADS,
        D_MODEL_K,
        BLOCK_HEAD=BLOCK_HEAD,
        BLOCK_DMODEL=BLOCK_DMODEL,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return
