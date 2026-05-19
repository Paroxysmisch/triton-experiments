import torch
import triton
import triton.language as tl


@triton.jit
def _fwd_kernel(
    Q, K, V, sm_scale, Out,
    stride_qbs, stride_qh, stride_qm, stride_qk,
    stride_kbs, stride_kh, stride_kn, stride_kk,
    stride_vbs, stride_vh, stride_vn, stride_vk,
    stride_obs, stride_oh, stride_om, stride_ok,
    Z,
    qk_type,
    kv_group_num: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    HAS_MASK: tl.constexpr,
    MASK_KV: tl.constexpr,
    MASK_BLOCK_N: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    HAS_KV_SPLIT: tl.constexpr,
    HAS_QK_SPLIT: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // kv_group_num
    off_h = off_hb % kv_group_num

    q_offset = ((off_b * stride_qbs) + (off_h * stride_qh))
    k_offset = ((off_b * stride_kbs) + (off_h * stride_kh))
    v_offset = ((off_b * stride_vbs) + (off_h * stride_vh))
    o_offset = ((off_b * stride_obs) + (off_h * stride_oh) +
                start_m * stride_om)

    q_ram_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(BLOCK_M, Q.shape[3]),
        strides=(stride_qm, stride_qk),
        offsets=(start_m, 0),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    k_ram_block_ptr = tl.make_block_ptr(
        base=K + k_offset,
        shape=(K.shape[1], K.shape[2]),
        strides=(stride_kn, stride_kk),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    v_ram_block_ptr = tl.make_block_ptr(
        base=V + v_offset,
        shape=(BLOCK_M, V.shape[3]),
        strides=(stride_vn, stride_vk),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    if HAS_KV_SPLIT:
        q_ram_block_ptr = tl.make_block_ptr(
            base=Q + q_offset,
            shape=(BLOCK_M, Q.shape[2]),
            strides=(stride_qm, stride_qk),
            offsets=(start_m, 0),
            block_shape=(BLOCK_M, BLOCK_N),
            order=(1, 0),
        )
        k_ram_block_ptr = tl.make_block_ptr(
            base=K + k_offset,
            shape=(K.shape[0], K.shape[2]),
            strides=(stride_kn, stride_kk),
            offsets=(0, 0),
            block_shape=(BLOCK_M, BLOCK_N),
            order=(1, 0),
        )
    q = tl.load(q_ram_block_ptr, boundary_check=(0, 1)).to(tl.float32)
    if BLOCK_HEADDIM == 1:
        q = tl.trans(q)

    if HAS_QK_SPLIT:
        k_ram_block_ptr = tl.make_block_ptr(
            base=K + k_offset,
            shape=(K.shape[0], K.shape[1]),
            strides=(stride_kn, stride_kk),
            offsets=(0, 0),
            block_shape=(BLOCK_N, BLOCK_M),
            order=(0, 1),
        )
    k = tl.load(k_ram_block_ptr, boundary_check=(0, 1)).to(tl.float32)
    if BLOCK_HEADDIM == 1:
        k = tl.trans(k)

    if HAS_KV_SPLIT:
        v_ram_block_ptr = tl.make_block_ptr(
            base=V + v_offset,
            shape=(BLOCK_N, V.shape[2]),
            strides=(stride_vk, stride_vn),
            offsets=(0, 0),
            block_shape=(BLOCK_N, BLOCK_M),
            order=(0, 1),
        )
    v = tl.load(v_ram_block_ptr, boundary_check=(0, 1)).to(tl.float32)
    if BLOCK_HEADDIM == 1:
        v = tl.trans(v)

    if MASK_KV:
        if MASK_BLOCK_N == 1:
            mask_offsets = tl.arange(0, BLOCK_M)
            mask_k_off = tl.make_block_ptr(
                base=Z + o_offset,
                shape=(BLOCK_M,),
                strides=(stride_qm,),
                offsets=(mask_offsets,),
                block_shape=(BLOCK_M,),
                order=(0,),
            )
            mask_k = tl.load(mask_k_off).to(tl.float32)
            mask_k = tl.expand_dims(mask_k, 1)
            k = tl.where(mask_k, k, float('-inf'))
        else:
            mask_offsets = tl.arange(0, BLOCK_M)[:, None]
            mask_v_off = tl.make_block_ptr(
                base=Z + o_offset,
                shape=(BLOCK_M, BLOCK_N),
                strides=(stride_om, stride_ok),
                offsets=(mask_offsets, 0),
                block_shape=(BLOCK_M, BLOCK_N),
                order=(0, 1),
            )
            mask_v = tl.load(mask_v_off).to(tl.float32)
            v = tl.where(mask_v, v, float('-inf'))

    if IS_CAUSAL:
        k = tl.where(
            tl.arange(0, BLOCK_M)[:, None] >= tl.arange(0, BLOCK_N)[None, :],
            k,
            float('-inf'),
        )

    if qk_type == 'float16':
        q = q.to(tl.float16)
        k = k.to(tl.float16)

    k = tl.trans(k)

    k_max = tl.max(k, 1)
    k_max = tl.broadcast(k_max, [BLOCK_M, BLOCK_N])
    k_shift = k - k_max
    k_shift = k_shift * sm_scale
    qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    if
