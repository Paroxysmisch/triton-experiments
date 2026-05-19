import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel(
    Q, K, V, sm_scale,  #
    Lq, Lk, Lv,  #
    M,  #
    Out,  #
    stride_qz, stride_qh, stride_qm, stride_qk,  #
    stride_kz, stride_kh, stride_kn, stride_kk,  #
    stride_vz, stride_vh, stride_vk, stride_vn,  #
    stride_oz, stride_oh, stride_om, stride_on,  #
    Z, H, N_CTX,  #
    IS_CAUSAL,  #
    USE_FP8,  #
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr,  #
    BLOCK_NUM: tl.constexpr,  #
    BLOCK_M_PADDED: tl.constexpr, BLOCK_N_PADDED: tl.constexpr,  #
):
    start_m = tl.program_id(0)
    off_z = tl.program_id(1)
    off_h = tl.program_id(2)
    off_bh = off_h * BLOCK_NUM + start_m // BLOCK_M
    offs_m = start_m + tl.arange(0, BLOCK_M_PADDED)
    offs_n = tl.arange(0, BLOCK_N_PADDED)
    z_range = off_z * BLOCK_DMODEL * 2
    q_ptrs = Q + z_range + off_h * stride_qh + offs_m[:, None] * stride_qm + offs_n[None, :] * stride_qk
    k_ptrs = K + z_range + off_h * stride_kh + offs_m[:, None] * stride_kn + offs_n[None, :] * stride_kk
    v_ptrs = V + z_range + off_h * stride_vh + offs_m[:, None] * stride_vk + offs_n[None, :] * stride_vn
    # mask to prevent attention to self and future words for causal attention
    upper_m = tl.where(off_bh == 0, N_CTX, 0)
    upper_n = tl.where(off_bh == 0, N_CTX, 0)
    mask_m = offs_m[:, None] < upper_m
    mask_n = offs_n[None, :] < upper_n
    if IS_CAUSAL:
        causal_mask_m = offs_m[:, None] >= upper_m
        mask_m = mask_m & causal_mask_m
    mask = mask_m & mask_n
    # load q, k, v
    if USE_FP8:
        # rematerialize offsets to save registers
        q_remat = tl.remat(q_ptrs, 0)
        k_remat = tl.remat(k_ptrs, 1)
        v_remat = tl.remat(v_ptrs, 2)
        q = tl.load(q_remat, mask=mask, other=0.0).to(tl.float32)
        k = tl.load(k_remat, mask=mask, other=0.0).to(tl.float32)
        v = tl.load(v_remat, mask=mask, other=0.0).to(tl.float32)
    else:
        q = tl.load(q_ptrs, mask=mask, other=0.0)
        k = tl.load(k_ptrs, mask=mask, other=0.0)
        v = tl.load(v_ptrs, mask=mask, other=0.0)
    # reshape
    q = tl.reshape(q, [BLOCK_M, BLOCK_DMODEL])
    k = tl.reshape(k, [BLOCK_N, BLOCK_DMODEL])
    v = tl.reshape(v, [BLOCK_N, BLOCK_DMODEL])
    # compute qk
    q = tl.trans(q)
    qk = tl.zeros([BLOCK_DMODEL, BLOCK_DMODEL], dtype=tl.float32)
    qk += tl.dot(q, k)
    # scale logits
    qk *= sm_scale
    # compute m
    m_i = tl.max(qk, 1)
    m_i = tl.expand_dims(m_i, 1)
    # subtract
    qk = qk - m_i
    # compute p
    p_i = tl.exp(qk)
    # mask
    p_i = tl.where(mask_m & mask_n, p_i, 0)
    # sum
    p_sum = tl.sum(p_i, 1)
    p_sum = tl.expand_dims(p_sum, 1)
    # compute o
    o = tl.dot(p_i, v)
    o /= p_sum
    # store o
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    o_ptrs = Out + z_range + off_h * stride_oh + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(o_ptrs, o.to(Out.dtype.element_ty), mask=mask)
    return

@torch.no_grad()
def triton_fa(q, k, v, o, q_first_dim_is_head, sm_scale, is_causal, use_fp8):
    # check data type consistency
    assert q.dtype == k.dtype and k.dtype == v.dtype and v.dtype == o.dtype
    assert q.dtype in [torch.float8_e4m3fn, torch.bfloat16, torch.float16, torch.float32]
    batch, head, seq_len, head_dim = q.shape
    assert head == k.shape[1] and head == v.shape[1] and head == o.shape[1]
    assert seq_len == o.shape[2]
    # configurations
    Lq, Lk, Lv = q.shape[2], k.shape[2], v.shape[2]
    assert q.stride(0) == 1 and k.stride(0) == 1 and v.stride(0) == 1 and o.stride(0) == 1
    assert q.stride(1) == 1 and k.stride(1) == 1 and v.stride(1) == 1 and o.stride(1) == 1
    assert q.stride(3) == 1 and k.stride(3) == 1 and v.stride(3) == 1 and o.stride(3) == 1
    assert q.stride(2) == q.stride(3) * q.dtype.itemsize
    assert k.stride(2) == k.stride(3) * k.dtype.itemsize
    assert v.stride(2) == v.stride(3) * v.dtype.itemsize
    assert o.stride(2) == o.stride(3) * o.dtype.itemsize
    # dynamic grid
    grid = lambda META: (triton.cdiv(seq_len, META["BLOCK_M"]), head, batch * head)  # noqa
    # triton kernel
    _fwd_kernel[grid](
        q, k, v, sm_scale,  #
        Lq, Lk, Lv,  #
        seq_len,
