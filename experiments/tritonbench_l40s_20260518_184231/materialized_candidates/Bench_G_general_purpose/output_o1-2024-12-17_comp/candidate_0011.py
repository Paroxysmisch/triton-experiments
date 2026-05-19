import triton
import triton.language as tl

@triton.jit
def rms_matmul_rbe(
    x_ptr,        # float*   [B, M, K]
    w_ptr,        # float*   [K, N] (transposed weight)
    rms_w_ptr,    # float*   [B] or [1] for RMS weight
    out_ptr,      # float*   [B, M, N]
    B, M, K, N,   # dimensions
    stride_xb, stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_rb,
    stride_ob, stride_om, stride_on,
    apply_rotary, # int (0 or 1)
    cos_ptr,      # optional float* (for rotary embedding)
    sin_ptr,      # optional float* (for rotary embedding)
    THETA,        # float (angle)
    BLOCK_SIZE_M: tl.constexpr, 
    BLOCK_SIZE_N: tl.constexpr, 
    BLOCK_SIZE_K: tl.constexpr
):
    # Program IDs
    b_idx = tl.program_id(0)
    m_block_idx = tl.program_id(1)

    # Offsets for block
    m_block_start = m_block_idx * BLOCK_SIZE_M
    m_range = tl.arange(0, BLOCK_SIZE_M)
    n_range = tl.arange(0, BLOCK_SIZE_N)
    k_range = tl.arange(0, BLOCK_SIZE_K)
    
    # Compute row indices for M dimension
    m_indices = m_block_start + m_range
    # Bound checking
    mask_m = m_indices < M

    # Pointers to input row and output
    x_row_ptr = x_ptr + b_idx * stride_xb + m_block_start * stride_xm
    out_row_ptr = out_ptr + b_idx * stride_ob + m_block_start * stride_om

    # -----------------------------
    # 1) RMS Normalization per row
    # -----------------------------
    # Load partial row [BLOCK_SIZE_M, K] in chunks
    # Accumulate sum of squares
    sq_sum = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32)
    for k_block_start in range(0, K, BLOCK_SIZE_K):
        k_indices = k_block_start + k_range
        mask_k = k_indices < K
        # Load x chunk
        x_val = tl.load(
            x_row_ptr + k_indices * stride_xk,
            mask=(mask_m[:, None] & mask_k[None, :]),
            other=0.0
        )
        sq_sum_chunk = tl.sum(x_val * x_val, axis=1)
        sq_sum += sq_sum_chunk

    # Compute RMS factor per row
    denom = 1.0 / tl.sqrt(sq_sum / tl.maximum(1, K))
    # Optionally multiply by the per-batch RMS weight
    rms_w_val = tl.load(rms_w_ptr + b_idx * stride_rb)
    denom = denom * rms_w_val

    # -----------------------------
    # 2) MatMul with optional Rotary
    # -----------------------------
    # We'll accumulate partial results for out row
    for n_block_start in range(0, N, BLOCK_SIZE_N):
        n_indices = n_block_start + n_range
        mask_n = n_indices < N
        acc = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)

        # Re-load x row + RMS normalization + rotary if needed
        # Then multiply by w in K dimension chunks
        for k_block_start in range(0, K, BLOCK_SIZE_K):
            k_indices = k_block_start + k_range
            mask_k = k_indices < K

            # Load x chunk
            x_val = tl.load(
                x_row_ptr + k_indices * stride_xk,
                mask=(mask_m[:, None] & mask_k[None, :]),
                other=0.0
            )
            # Normalize
            x_val = x_val * denom[:, None]

            # If apply_rotary != 0, do rotary embedding
            if apply_rotary != 0:
                # For illustration, assume half-rotary on dimension
                half_k = K // 2
                # Indices within half
                x_left = tl.multiple_of(x_val[:, :half_k], BLOCK_SIZE_K)
                x_right = tl.multiple_of(x_val[:, half_k:], BLOCK_SIZE_K)
                # load cos, sin from pointers if needed
                cos_v = 1.0
                sin_v = 0.0
                # option: use THETA or cos/sin array
                if cos_ptr and sin_ptr:
                    cos_v = tl.load(cos_ptr + m_indices, mask=mask_m, other=1.0)
                    sin_v = tl.load(sin_ptr + m_indices, mask=mask_m, other=0.0)
                # rotate
                # just a single value approach for demonstration
                x_val_left = x_left * cos_v[:, None] - x_right * sin_v[:, None]
                x_val_right = x_left * sin_v[:, None] + x_right * cos_v[:, None]
                x_val = tl.cat([x_val_left, x_val_right], 1)
            
            # Load weight chunk
            w_val = tl.load(
                w_ptr + k_indices[:, None] * stride_wk + n_indices[None, :] * stride_wn,
                mask=(mask_k[:, None] & mask_n[None, :]),
                other=0.0
            )
            # Accumulate
            acc += tl.dot(x_val, w_val)

        # Store partial results
        tl.store(
            out_row_ptr + n_indices[None, :] * stride_on,
            acc,
            mask=(mask_m[:, None] & mask_n[None, :])
        )


def rms_matmul_rbe_wrapper(
    x, w, rms_w,
    apply_rotary=False,
    cos=None,
    sin=None,
    THETA=0.0
):
    """
    x:  [B, M, K]  (float32)
    w:  [K, N]     (float32, transposed weight)
    rms_w: [B] or [1]
    cos, sin: optional buffers for rotary embedding
    """
    B, M, K = x.shape
    N = w.shape[1]
    out = x.new_empty((B, M, N))

    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32

    grid = (B, (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M)
    triton.run(
        rms_matmul_rbe,
        grid=grid,
        num_warps=4,
        num_stages=2,
        args=[
            x, w, rms_w, out,
            B, M, K, N,
            x.stride(0), x.stride(1), x.stride(2),
            w.stride(0), w.stride(1),
            rms_w.stride(0),
            out.stride(0), out.stride(1), out.stride(2),
            int(apply_rotary),
            cos if cos is not None else 0,
            sin if sin is not None else 0,
            THETA,
        ],
        constants={
            "BLOCK_SIZE_M": BLOCK_SIZE_M,
            "BLOCK_SIZE_N": BLOCK_SIZE_N,
            "BLOCK_SIZE_K": BLOCK_SIZE_K,
        }
    )
    return out
