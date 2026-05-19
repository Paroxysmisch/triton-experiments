import torch
import triton
import triton.language as tl

@triton.jit
def decoding_fused_rotary_embedding_kernel(
    q, q_rot, q_rot_s, q_h_s, q_t_s, q_dim_s, q_head_num, q_total_tokens,
    k_cache, k_cache_s, k_h_s, k_t_s, k_dim_s, k_head_num, k_cache_len,
    v_cache, v_cache_s, v_h_s, v_t_s, v_dim_s, v_head_num, v_cache_len,
    kv_tables, kv_tables_s, kv_tables_h_s, kv_tables_t_s,
    kv_lengths, kv_lengths_s, kv_lengths_h_s,
    cos, sin,
    start_token_id,
    max_input_len,
    KV_GROUP_NUM: tl.constexpr,
    BLOCK: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    use_new_kcache_layout: tl.constexpr,
):
    # Get program IDs
    q_head_idx = tl.program_id(axis=0)
    q_token_idx = tl.program_id(axis=1)
    kv_group_idx = tl.program_id(axis=2)

    # Get offsets for current program
    q_head_off = q_head_idx * q_h_s
    q_token_off = q_token_idx * q_t_s
    q_dim_off = tl.arange(0, BLOCK) * q_dim_s

    kv_group_len = tl.load(kv_lengths + kv_lengths_h_s * kv_group_idx + kv_lengths_s)
    if kv_group_idx == KV_GROUP_NUM - 1 and q_token_idx + max_input_len > kv_cache_len:
        kv_group_len -= (q_token_idx + max_input_len) - kv_cache_len

    kv_token_idx = tl.arange(0, BLOCK)
    if IS_CAUSAL:
        kv_token_idx += (q_token_idx + start_token_id)

    kv_head_idx = tl.load(kv_tables + kv_tables_h_s * kv_group_idx + kv_tables_t_s * kv_token_idx)

    # Get offsets for k_cache
    if use_new_kcache_layout:
        k_head_off = kv_head_idx * k_h_s
        k_token_off = kv_token_idx * k_t_s
    else:
        k_head_off = kv_head_idx * k_h_s
        k_token_off = 0

    # Get offsets for v_cache
    if use_new_kcache_layout:
        v_head_off = kv_head_idx * v_h_s
        v_token_off = kv_token_idx * v_t_s
    else:
        v_head_off = kv_head_idx * v_h_s
        v_token_off = 0

    # Load q
    q_ptrs = q + q_head_off + q_token_off + q_dim_off
    q_rot_ptrs = q_rot + q_head_off + q_token_off + q_dim_off
    q_dim_mask = q_dim_off < q_dim_s

    # Load cos/sin
    cos_ptrs = cos + q_head_off + q_token_off + q_dim_off
    sin_ptrs = sin + q_head_off + q_token_off + q_dim_off

    # Load k_cache
    k_cache_ptrs = k_cache + k_cache_s * k_head_off + k_cache_s * k_token_off + q_dim_off

    # Load v_cache
    v_cache_ptrs = v_cache + v_cache_s * v_head_off + v_cache_s * v_token_off + q_dim_off

    # Compute sin/cos
    dim_mask = q_dim_off < q_dim_s
    q_dim_half = q_dim_s // 2
    q_dim_off_half = q_dim_off[None, :] - q_dim_half
    q_dim_off_half_sin = q_dim_off_half * (2 * 3.14159265 / q_dim_half)
    q_dim_off_half_cos = q_dim_off_half * (2 * 3.14159265 / q_dim_half)
    sin_val = tl.load(sin_ptrs, mask=dim_mask) * q_dim_off_half_sin
    cos_val = tl.load(cos_ptrs, mask=dim_mask) * q_dim_off_half_cos

    # Apply rotary
    q_rot_val = tl.load(q_ptrs, mask=q_dim_mask, other=0) * cos_val - tl.load(q_ptrs + q_dim_half, mask=q_dim_mask, other=0) * sin_val
    tl.store(q_rot_ptrs, q_rot_val, mask=q_dim_mask)

    # Store back to q
    tl.store(q_ptrs, q_rot_val, mask=q_dim_mask)
    tl.store(q_ptrs + q_dim_half, tl.load(q_ptrs + q_dim_half, mask=q_dim_mask, other=0) * cos_val + q_rot_val * sin_val, mask=q_dim_mask)

    # Store k_cache
    tl.store(k_cache_ptrs, tl.load(k_cache_ptrs, mask=dim_mask, other=0), mask=dim_mask)

    # Store v_cache
    tl.store(v_cache_ptrs, tl.load(v_cache_ptrs, mask=dim_mask, other=0), mask=dim_mask)

    return

@torch.inference_mode()
def decoding_fused_rotary_embedding(
    q: torch.Tensor,
    q_rot: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    kv_tables: torch.Tensor,
    kv_lengths: torch.Tensor,
    start_token_id: int,
    max_input_len: int,
    KV_GROUP_NUM: int,
    BLOCK: int,
    IS_CAUSAL: bool,
    use_new_kcache_layout: bool,
    block_tables: torch.Tensor,
):
    # q: [batch, head, seq_len, dim]
    # q_rot: [batch, head, seq_len, dim]
    # cos: [batch, head, seq_len, dim // 2]
    # sin: [batch, head, seq_len, dim // 2]
    # k_cache: [batch, head, kv_cache_len, dim]
    # v_cache: [batch, head, kv_cache_len, dim]
    # kv_tables: [batch, head, seq_len]
    # kv_lengths: [batch, head]
    # block_tables: [2, batch, head, seq_len]

    batch, head_num, q_len, dim = q.shape
    _, _, kv_cache_len, _ = k_cache.shape
    head_dim = dim // head_num
    assert cos.shape == q.shape
    assert sin.
