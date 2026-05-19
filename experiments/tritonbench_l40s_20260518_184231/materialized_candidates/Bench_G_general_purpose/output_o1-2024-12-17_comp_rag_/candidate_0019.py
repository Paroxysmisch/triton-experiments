import torch
import triton
import triton.language as tl


@triton.jit
def rotary_embedding_kernel(
    Q, K, COS, SIN, Q_EMB, K_EMB,
    seq_len,
    stride_qs: tl.constexpr, stride_qh: tl.constexpr, stride_qd: tl.constexpr,
    stride_ks: tl.constexpr, stride_kh: tl.constexpr, stride_kd: tl.constexpr,
    stride_qes: tl.constexpr, stride_qeh: tl.constexpr, stride_qed: tl.constexpr,
    stride_kes: tl.constexpr, stride_keh: tl.constexpr, stride_ked: tl.constexpr,
    half_size: tl.constexpr, BLOCK: tl.constexpr,
    BLOCK_QH: tl.constexpr, BLOCK_N: tl.constexpr
):
    seq_block_id = tl.program_id(0)
    head_id = tl.program_id(1)

    pos_offset = seq_block_id * BLOCK + tl.arange(0, BLOCK)
    pos_mask = pos_offset < seq_len
    pos_offset = tl.max_contiguous(pos_offset % seq_len, BLOCK)

    feat_size = half_size * 2
    feat_offset_l = tl.arange(0, BLOCK_N)
    feat_mask = feat_offset_l < half_size
    feat_offset_l = feat_offset_l % half_size
    feat_offset_h = half_size + feat_offset_l
    seq_mask = pos_mask[:, None] & feat_mask[None, :]

    cs_offset_l = pos_offset[:, None] * feat_size + feat_offset_l[None, :]
    cs_offset_h = pos_offset[:, None] * feat_size + feat_offset_h[None, :]
    q_elem_type = Q.dtype.element_ty

    cos_l = tl.load(COS + cs_offset_l).to(q_elem_type)
    cos_h = tl.load(COS + cs_offset_h).to(q_elem_type)
    sin_l = tl.load(SIN + cs_offset_l).to(q_elem_type)
    sin_h = tl.load(SIN + cs_offset_h).to(q_elem_type)

    if head_id < BLOCK_QH:
        q_ptr = Q + pos_offset * stride_qs
        qe_ptr = Q_EMB + pos_offset * stride_qes
        ql_ptrs = q_ptr[:, None] + feat_offset_l[None, :] * stride_qd
        qh_ptrs = q_ptr[:, None] + feat_offset_h[None, :] * stride_qd
        qel_ptrs = qe_ptr[:, None] + feat_offset_l[None, :] * stride_qed
        qeh_ptrs = qe_ptr[:, None] + feat_offset_h[None, :] * stride_qed
        ql_ptrs += head_id * stride_qh
        qh_ptrs += head_id * stride_qh
        qel_ptrs += head_id * stride_qeh
        qeh_ptrs += head_id * stride_qeh

        q_l = tl.load(ql_ptrs)
        q_h = tl.load(qh_ptrs)
        qe_l = q_l * cos_l - q_h * sin_l
        qe_h = q_h * cos_h + q_l * sin_h

        tl.store(qel_ptrs, qe_l, mask=seq_mask)
        tl.store(qeh_ptrs, qe_h, mask=seq_mask)
    else:
        head_id = head_id - BLOCK_QH
        k_ptr = K + pos_offset * stride_ks
        ke_ptr = K_EMB + pos_offset * stride_kes
        kl_ptrs = k_ptr[:, None] + feat_offset_l[None, :] * stride_kd
        kh_ptrs = k_ptr[:, None] + feat_offset_h[None, :] * stride_kd
        kel_ptrs = ke_ptr[:, None] + feat_offset_l[None, :] * stride_ked
        keh_ptrs = ke_ptr[:, None] + feat_offset_h[None, :] * stride_ked
        kl_ptrs += head_id * stride_kh
        kh_ptrs += head_id * stride_kh
        kel_ptrs += head_id * stride_keh
        keh_ptrs += head_id * stride_keh

        k_l = tl.load(kl_ptrs)
        k_h = tl.load(kh_ptrs)
        ke_l = k_l * cos_l - k_h * sin_l
        ke_h = k_h * cos_h + k_l * sin_h

        tl.store(kel_ptrs, ke_l, mask=seq_mask)
        tl.store(keh_ptrs, ke_h, mask=seq_mask)


@triton.jit
def fused_rotary_embedding_kernel_v2(
    Q, K, COS, SIN, Q_EMB, K_EMB,
    K_CACHE, BLOCK_TABLES, KV_LENGTHS,
    seq_len,
    stride_qs: tl.constexpr, stride_qh: tl.constexpr, stride_qd: tl.constexpr,
    stride_ks: tl.constexpr, stride_kh: tl.constexpr, stride_kd: tl.constexpr,
    stride_qes: tl.constexpr, stride_qeh: tl.constexpr, stride_qed: tl.constexpr,
    stride_kes: tl.constexpr, stride_keh: tl.constexpr, stride_ked: tl.constexpr,
    stride_cache_s: tl.constexpr, stride_cache_h: tl.constexpr, stride_cache_d: tl.constexpr,
    half_size: tl.constexpr, BLOCK: tl.constexpr,
    BLOCK_QH: tl.constexpr, BLOCK_N: tl.constexpr
):
    seq_block_id = tl.program_id(0)
    head_id = tl.program_id(1)

    pos_offset = seq_block_id * BLOCK + tl.arange(0, BLOCK)
    pos_mask = pos_offset < seq_len
    pos_offset = tl.max_contiguous(pos_offset % seq_len, BLOCK)

    feat_size = half_size * 2
    feat_offset_l = tl.arange(0, BLOCK_N)
    feat_mask = feat_offset_l < half_size
    feat_offset_l = feat_offset_l % half_size
    feat_offset_h = half_size + feat_offset_l
    seq_mask = pos_mask[:, None] & feat_mask[None, :]

    cs_offset_l = pos_offset[:, None] * feat_size + feat_offset_l[None, :]
    cs_offset_h = pos_offset[:, None] * feat_size + feat_offset_h[None, :]
    q_elem_type = Q.dtype.element_ty

    cos_l = tl.load(COS + cs_offset_l).to(q_elem_type)
    cos_h = tl.load(COS + cs_offset_h).to(q_elem_type)
    sin_l = tl.load(SIN + cs_offset_l).to(q_elem_type)
    sin_h = tl.load(SIN + cs_offset_h).to(q_elem_type)

    q_ptr = Q + pos_offset * stride_qs
    qe_ptr = Q_EMB + pos_offset * stride_qes
    k_ptr = K + pos_offset * stride_ks
    ke_ptr = K_EMB + pos_offset * stride_kes

    if head_id < BLOCK_QH:
        ql_ptrs = q_ptr[:, None] + feat_offset_l[None, :] * stride_qd + head_id * stride_qh
        qh_ptrs = q_ptr[:, None] + feat_offset_h[None, :] * stride_qd + head_id * stride_qh
        qel_ptrs = qe_ptr[:, None] + feat_offset_l[None, :] * stride_qed + head_id * stride_qeh
        qeh_ptrs = qe_ptr[:, None] + feat_offset_h[None, :] * stride_qed + head_id * stride_qeh

        q_l = tl.load(ql_ptrs)
        q_h = tl.load(qh_ptrs)
        qe_l = q_l * cos_l - q_h * sin_l
        qe_h = q_h * cos_h + q_l * sin_h

        tl.store(qel_ptrs, qe_l, mask=seq_mask)
        tl.store(qeh_ptrs, qe_h, mask=seq_mask)
    else:
        head_id = head_id - BLOCK_QH
        kl_ptrs = k_ptr[:, None] + feat_offset_l[None, :] * stride_kd + head_id * stride_kh
        kh_ptrs = k_ptr[:, None] + feat_offset_h[None, :] * stride_kd + head_id * stride_kh
        kel_ptrs = ke_ptr[:, None] + feat_offset_l[None, :] * stride_ked + head_id * stride_keh
        keh_ptrs = ke_ptr[:, None] + feat_offset_h[None, :] * stride_ked + head_id * stride_keh

        k_l = tl.load(kl_ptrs)
        k_h = tl.load(kh_ptrs)
        ke_l = k_l * cos_l - k_h * sin_l
        ke_h = k_h * cos_h + k_l * sin_h

        tl.store(kel_ptrs, ke_l, mask=seq_mask)
        tl.store(keh_ptrs, ke_h, mask=seq_mask)

        # Block-based cache store
        block_id = tl.load(BLOCK_TABLES + pos_offset)
        kv_len = tl.load(KV_LENGTHS)
        # Compute the offset in the cache (simplified example)
        cache_offset = block_id * stride_cache_s + head_id * stride_cache_h
        kl_cache_ptrs = K_CACHE + cache_offset + feat_offset_l[None, :] * stride_cache_d
        kh_cache_ptrs = K_CACHE + cache_offset + feat_offset_h[None, :] * stride_cache_d

        tl.store(kl_cache_ptrs, ke_l, mask=seq_mask & (pos_offset[:, None] < kv_len))
        tl.store(kh_cache_ptrs, ke_h, mask=seq_mask & (pos_offset[:, None] < kv_len))


def rotary_embedding(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    q_embed: torch.Tensor = None,
    k_embed: torch.Tensor = None,
    k_cache: torch.Tensor = None,
    block_tables: torch.Tensor = None,
    kv_lengths: torch.Tensor = None
):
    if cos.device != q.device:
        cos = cos.to(device=q.device)
    if sin.device != q.device:
        sin = sin.to(device=q.device)
    if q_embed is None:
        q_embed = torch.empty_like(q)
    if k_embed is None:
