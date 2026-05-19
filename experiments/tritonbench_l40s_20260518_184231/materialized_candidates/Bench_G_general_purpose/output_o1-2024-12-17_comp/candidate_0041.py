import torch
import triton
import triton.language as tl

@triton.jit
def _triton_rope(
    Q_PTR, K_PTR,
    COS_PTR, SIN_PTR,
    Q_ROW_STRIDE, K_ROW_STRIDE,
    COS_STRIDE, SIN_STRIDE,
    HEAD_DIM, BACKWARD_PASS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # program_id ranges over batch * seq
    pid = tl.program_id(0)

    # Offsets in Q/K space
    q_off = pid * Q_ROW_STRIDE
    k_off = pid * K_ROW_STRIDE

    # We only rotate half the dimension at a time
    half_dim = HEAD_DIM // 2

    # Indices for the first half (even) and second half (odd)
    idx = tl.arange(0, BLOCK_SIZE)
    even_mask = idx < half_dim
    odd_mask = (idx >= half_dim) & (idx < HEAD_DIM)

    # Compute final in-bounds
    in_bounds_even = even_mask & (idx < HEAD_DIM)
    in_bounds_odd = odd_mask & (idx < HEAD_DIM)

    # Load Q, K slices
    q_val = tl.load(Q_PTR + q_off + idx, mask=idx < HEAD_DIM, other=0.0)
    k_val = tl.load(K_PTR + k_off + idx, mask=idx < HEAD_DIM, other=0.0)

    # Load cos, sin slices
    cos_val = tl.load(COS_PTR + pid * COS_STRIDE + idx, mask=idx < HEAD_DIM, other=1.0)
    sin_val = tl.load(SIN_PTR + pid * SIN_STRIDE + idx, mask=idx < HEAD_DIM, other=0.0)

    # Split Q, K into even/odd halves
    q_even = tl.where(in_bounds_even, q_val, 0.0)
    q_odd  = tl.where(in_bounds_odd,  q_val, 0.0)
    k_even = tl.where(in_bounds_even, k_val, 0.0)
    k_odd  = tl.where(in_bounds_odd,  k_val, 0.0)

    # For forward pass:
    #   q_even' = q_even*cos - q_odd*sin
    #   q_odd'  = q_even*sin + q_odd*cos
    # For backward pass:
    #   q_even' = q_even*cos + q_odd*sin
    #   q_odd'  = -q_even*sin + q_odd*cos

    if BACKWARD_PASS == 0:
        # forward
        new_q_even = q_even * cos_val - q_odd * sin_val
        new_q_odd  = q_even * sin_val + q_odd * cos_val

        new_k_even = k_even * cos_val - k_odd * sin_val
        new_k_odd  = k_even * sin_val + k_odd * cos_val
    else:
        # backward
        new_q_even = q_even * cos_val + q_odd * sin_val
        new_q_odd  = -q_even * sin_val + q_odd * cos_val

        new_k_even = k_even * cos_val + k_odd * sin_val
        new_k_odd  = -k_even * sin_val + k_odd * cos_val

    # Merge back
    out_q_val = tl.where(in_bounds_even, new_q_even, 0.0) + tl.where(in_bounds_odd, new_q_odd, 0.0)
    out_k_val = tl.where(in_bounds_even, new_k_even, 0.0) + tl.where(in_bounds_odd, new_k_odd, 0.0)

    # Store
    tl.store(Q_PTR + q_off + idx, out_q_val, mask=idx < HEAD_DIM)
    tl.store(K_PTR + k_off + idx, out_k_val, mask=idx < HEAD_DIM)


def rope_forward(q, k, cos, sin):
    """
    Applies rotary position embedding (RoPE) transformations to q and k
    using cos and sin. Operates on (B, S, H, D) shaped tensors, but
    flattens them to batch*seq rows for efficient Triton processing.
    """
    B, S, H, D = q.shape

    # Transpose/reshape for contiguous memory if needed
    q_ = q.view(B * S, H * D).contiguous()
    k_ = k.view(B * S, H * D).contiguous()
    cos_ = cos.view(B * S, H * D).contiguous()
    sin_ = sin.view(B * S, H * D).contiguous()

    # Strides
    q_row_stride = q_.stride(0)
    k_row_stride = k_.stride(0)
    cos_stride = cos_.stride(0)
    sin_stride = sin_.stride(0)

    # Launch grid: one block per (batch, sequence) pair
    grid = (B * S,)

    # Choose block size (must not exceed head dim)
    BLOCK_SIZE = min(D * H, 128)

    _triton_rope[grid](
        q_, k_,
        cos_, sin_,
        q_row_stride, k_row_stride,
        cos_stride, sin_stride,
        H * D,  # HEAD_DIM
        BACKWARD_PASS=0,  # Forward pass
        BLOCK_SIZE=BLOCK_SIZE
    )

    # Reshape back to original
    q_out = q_.view(B, S, H, D)
    k_out = k_.view(B, S, H, D)
    cos_out = cos_.view(B, S, H, D)
    sin_out = sin_.view(B, S, H, D)
    return q_out, k_out, cos_out, sin_out
