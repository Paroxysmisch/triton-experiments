import triton
import triton.language as tl

@triton.jit
def _copy_to_kcache_seqlen_n_kernel(
    k_ptr,                     # pointer to input k
    kcache_ptr,                # pointer to kcache
    seqlens_ptr,               # pointer to seqlens array
    batch_size,                # total batch size
    num_heads,                 # number of attention heads
    stride_bs_k,               # stride for batch dimension in k
    stride_bh_k,               # stride for head dimension in k
    stride_bhs_k,              # stride for seq_len dimension in k
    stride_bhsl_k,             # stride for head_dim dimension in k
    stride_kcache_b,           # stride for batch dimension in kcache
    stride_kcache_h,           # stride for head dimension in kcache
    stride_kcache_m,           # stride for "M" dimension in kcache
    stride_kcache_x,           # stride for "X" dimension in kcache
    HEAD_DIM: tl.constexpr,    # head dimension
    KCACHE_X: tl.constexpr,    # block size for partial head dimension
    IS_NEW_FORMAT: tl.constexpr  # flag to indicate new format or not
):
    # program_id(0) -> batch index
    b_id = tl.program_id(0)
    # program_id(1) -> head index
    h_id = tl.program_id(1)

    # Current token index based on seqlens
    curr_token = tl.load(seqlens_ptr + b_id)
    # (Optional) For new format, might adjust curr_token differently
    if IS_NEW_FORMAT:
        curr_token = curr_token + 0  # placeholder for possible offset adjustments

    # We will read HEAD_DIM values of K for this token and store them into kcache
    # We'll do a vectorized load/store along X dimension
    # logical range is HEAD_DIM, but we block in multiples of KCACHE_X
    block_offsets = tl.arange(0, KCACHE_X)
    # If HEAD_DIM is not a multiple of KCACHE_X, we mask out-of-bounds
    head_mask = block_offsets < HEAD_DIM

    # Base pointers for reading from k
    # k layout: [b, h, seq, d]
    # offset for the correct batch b_id, head h_id, position curr_token
    k_read_offset = (b_id * stride_bs_k
                     + h_id * stride_bh_k
                     + curr_token * stride_bhs_k)

    # Base pointers for writing to kcache
    # Suppose kcache layout: [b, h, M, X]
    # M dimension might be the token dimension in blocks, or col dimension
    # offset = b_id * stride_kcache_b + h_id * stride_kcache_h + ...
    # For demonstration, we treat curr_token as "M" index
    # We store along X dimension
    k_write_offset = (b_id * stride_kcache_b
                      + h_id * stride_kcache_h
                      + curr_token * stride_kcache_m)

    # Load from k
    k_vals = tl.load(k_ptr + k_read_offset + block_offsets * stride_bhsl_k, mask=head_mask, other=0.0)

    # Store to kcache
    tl.store(kcache_ptr + k_write_offset + block_offsets * stride_kcache_x, k_vals, mask=head_mask)


def copy_k_to_blocked_cache(k, kcache, seqlens, HEAD_DIM, KCACHE_X, is_new_format):
    """
    Copy 4D tensor k [batch, heads, seqlen, head_dim]
    into kcache, which has a blocked structure.
    seqlens is [batch] array containing the current token index per sequence.
    HEAD_DIM is the head dimension.
    KCACHE_X is the tile size used for partial 'head_dim' blocking.
    is_new_format is a boolean controlling layout differences.
    """
    # Extract necessary shapes and strides
    b, h, s, d = k.shape
    # Strides for k (assuming row-major like PyTorch)
    stride_bs_k   = k.stride(0)
    stride_bh_k   = k.stride(1)
    stride_bhs_k  = k.stride(2)
    stride_bhsl_k = k.stride(3)

    # For kcache, we assume shape [b, h, M, X] or something similar
    # Strides for kcache
    stride_kcache_b = kcache.stride(0)
    stride_kcache_h = kcache.stride(1)
    stride_kcache_m = kcache.stride(2)
    stride_kcache_x = kcache.stride(3)

    # Grid: (batch_size, num_heads)
    grid = (b, h)

    # Launch Triton kernel
    _copy_to_kcache_seqlen_n_kernel[grid](
        k,                      # k_ptr
        kcache,                 # kcache_ptr
        seqlens,                # seqlens_ptr
        b,                      # batch_size
        h,                      # num_heads
        stride_bs_k,
        stride_bh_k,
        stride_bhs_k,
        stride_bhsl_k,
        stride_kcache_b,
        stride_kcache_h,
        stride_kcache_m,
        stride_kcache_x,
        HEAD_DIM,
        KCACHE_X,
        is_new_format
    )
