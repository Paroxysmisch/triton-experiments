import torch
import triton
import triton.language as tl

# Constants
Q_HEAD_NUM = 8  # Number of query heads
HEAD_DIM = 64   # Dimension of each head

# Kernel for rotary embedding without k_cache
@triton.jit
def rotary_embedding_kernel(
    Q,
    K,
    COS,
    SIN,
    seq_len,
    stride_qs: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_ks: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_kd: tl.constexpr,
    half_size: tl.constexpr,
    BLOCK: tl.constexpr,
    BLOCK_QH: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    seq_block_id = tl.program_id(0)
    head_id = tl.program_id(1)

    pos_offset = seq_block_id * BLOCK + tl.arange(0, BLOCK)
    pos_mask = pos_offset < seq_len
    pos_offset = tl.max_contiguous(tl.multiple_of(pos_offset % seq_len, BLOCK), BLOCK)

    feat_size = half_size * 2
    feat_offset_l = tl.arange(0, BLOCK_N)
    feat_mask = feat_offset_l < half_size
    feat_offset_l = feat_offset_l % half_size
    feat_offset_h = half_size + feat_offset_l
    seq_mask = pos_mask[:, None] and feat_mask[None, :]
    cs_offset_l = pos_offset[:, None] * feat_size + feat_offset_l[None, :]
    cs_offset_h = pos_offset[:, None] * feat_size + feat_offset_h[None, :]
    q_elem_type = Q.dtype.element_ty
    cos_l = tl.load(COS + cs_offset_l).to(q_elem_type)
    cos_h = tl.load(COS + cs_offset_h).to(q_elem_type)
    sin_l = tl.load(SIN + cs_offset_l).to(q_elem_type)
    sin_h = tl.load(SIN + cs_offset_h).to(q_elem_type)

    if head_id < BLOCK_QH:
        q_ptr = Q + pos_offset * stride_qs
        ql_ptrs = q_ptr[:, None] + feat_offset_l[None, :] * stride_qd
        qh_ptrs = q_ptr[:, None] + feat_offset_h[None, :] * stride_qd
        ql_ptrs += head_id * stride_qh
        qh_ptrs += head_id * stride_qh

        q_l = tl.load(ql_ptrs)
        q_h = tl.load(qh_ptrs)
        qe_l = q_l * cos_l - q_h * sin_l
        qe_h = q_h * cos_h + q_l * sin_h

        tl.store(ql_ptrs, qe_l, mask=seq_mask)
        tl.store(qh_ptrs, qe_h, mask=seq_mask)
    else:
        head_id = head_id - BLOCK_QH
        k_ptr = K + pos_offset * stride_ks
        kl_ptrs = k_ptr[:, None] + feat_offset_l[None, :] * stride_kd
        kh_ptrs = k_ptr[:, None] + feat_offset_h[None, :] * stride_kd
        kl_ptrs += head_id * stride_kh
        kh_ptrs += head_id * stride_kh
        k_l = tl.load(kl_ptrs)
        k_h = tl.load(kh_ptrs)
        ke_l = k_l * cos_l - k_h * sin_l
        ke_h = k_h * cos_h + k_l * sin_h

        tl.store(kl_ptrs, ke_l, mask=seq_mask)
        tl.store(kh_ptrs, ke_h, mask=seq_mask)

# Kernel for fused rotary embedding with k_cache
@triton.jit
def fused_rotary_embedding_kernel_v2(
    Q,
    K,
    K_CACHE,
    BLOCK_TABLES,
    KV_LENGTHS,
    COS,
    SIN,
    seq_len,
    stride_qs: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_ks: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_kd: tl.constexpr,
    stride_kc: tl.constexpr,
    stride_kcs: tl.constexpr,
    stride_kch: tl.constexpr,
    stride_kcd: tl.constexpr,
    half_size: tl.constexpr,
    BLOCK: tl.constexpr,
    BLOCK_QH: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    seq_block_id = tl.program_id(0)
    head_id = tl.program_id(1)

    pos_offset = seq_block_id * BLOCK + tl.arange(0, BLOCK)
    pos_mask = pos_offset < seq_len
    pos_offset = tl.max_contiguous(tl.multiple_of(pos_offset % seq_len, BLOCK), BLOCK)

    feat_size = half_size * 2
    feat_offset_l = tl.arange(0, BLOCK_N)
    feat_mask = feat_offset_l < half_size
    feat_offset_l = feat_offset_l % half_size
    feat_offset_h = half_size + feat_offset_l
    seq_mask = pos_mask[:, None] and feat_mask[None, :]
    cs_offset_l = pos_offset[:, None] * feat_size + feat_offset_l[None, :]
    cs_offset_h = pos_offset[:, None] * feat_size + feat_offset_h[None, :]
    q_elem_type = Q.dtype.element_ty
    cos_l = tl.load(COS + cs_offset_l).to(q_elem_type)
    cos_h = tl.load(COS + cs_offset_h).to(q_elem_type)
    sin_l = tl.load(SIN + cs_offset_l).to(q_elem_type)
    sin_h = tl.load(SIN + cs_offset_h).to(q_elem_type)

    if head_id < BLOCK_QH:
        q_ptr = Q + pos_offset * stride_qs
        ql_ptrs = q_ptr[:, None] + feat_offset_l[None, :] * stride_qd
        qh_ptrs = q_ptr[:, None] + feat_offset_h[None, :] * stride_qd
        ql_ptrs += head_id * stride_qh
        qh_ptrs += head_id * stride_qh

        q_l = tl.load(ql_ptrs)
        q_h = tl.load(qh_ptrs)
        qe_l = q_l * cos_l - q_h * sin_l
        qe_h = q_h * cos_h + q_l * sin_h

        tl.store(ql_ptrs, qe_l, mask=seq_mask)
        tl.store(qh_ptrs, qe_h, mask=seq_mask)
    else:
        head_id = head_id - BLOCK_QH
        k_ptr = K + pos_offset * stride_ks
        k_cache_ptr = K_CACHE + pos_offset * stride_kcs
        kl_ptrs = k_ptr[:, None] + feat_offset_l[None, :] * stride_kd
        kh_ptrs = k_ptr[:, None] + feat_offset_h[None, :] * stride_kd
        kcl_ptrs = k_cache_ptr[:, None] + feat_offset_l[None, :] * stride_kcd
        kch_ptrs = k_cache_ptr[:, None] + feat_offset_h[None, :] * stride_kcd
        kl_ptrs += head_id * stride_kh
        kh_ptrs += head_id * stride_kh
        kcl_ptrs += head_id * stride_kch
        kch_ptrs += head_id * stride_kch
        k_l = tl.load(kl_ptrs)
        k_h = tl.load(kh_ptrs)
        ke_l = k_l * cos_l - k_h * sin_l
        ke_h = k_h * cos_h + k_l * sin_h

        tl.store(kl_ptrs, ke_l, mask=seq_mask)
        tl.store(kh_ptrs, ke_h, mask=seq_mask)
        tl.store(kcl_ptrs, ke_l, mask=seq_mask)
        tl.store(kch_ptrs, ke_h, mask=seq_mask)

# Wrapper function
def apply_rotary_pos_emb(q: torch.Tensor,
                         k: torch.Tensor,
                         cos: torch.Tensor,
                         sin: torch.Tensor,
                         k_cache: torch.Tensor = None,
                         block_tables: torch.Tensor = None,
                         kv_lengths: torch.Tensor = None):
    """Apply rotary positional embedding on query and key.

    Args:
        q (Tensor): Query state.
        k (Tensor): Key state.
        cos (Tensor): cosine matrix (seq_len, dim).
        sin (Tensor): sine matrix (seq_len, dim).
        k_cache (Tensor, optional): Key cache tensor. Defaults to None.
        block_tables (Tensor, optional): Block tables for cache management. Defaults to None.
        kv_lengths (Tensor, optional): Past sequence length information. Defaults to None.

    Returns:
        Tuple[Tensor, Tensor]: Embedded query and key.
    """
    if cos.device != q.device:
        cos = cos.to(device=q.device)
    if sin.device != q.device:
        sin = sin.to(device=q.device)

    seq_len = cos.numel() // cos.size(-1)
    half_size = q.size(-1) // 2
    BLOCK = 16
    BLOCK_N = triton.next_power_of_2(half_size)
    num_heads_q = q.size
