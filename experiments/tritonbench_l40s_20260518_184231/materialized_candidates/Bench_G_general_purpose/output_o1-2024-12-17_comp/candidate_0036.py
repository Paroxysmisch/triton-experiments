import triton
import triton.language as tl

@triton.jit
def _score_kernel(
    Q_ptr,  # [batch, heads, N_CTX_Q, dim]
    K_ptr,  # [batch, heads, dim, N_CTX_K]
    M_ptr,  # [batch, heads, N_CTX_Q, N_CTX_K] or broadcastable mask
    Out_ptr, # [batch, heads, N_CTX_Q, N_CTX_K]
    stride_qbh, stride_qbd, stride_qm,         # Q strides: batch-head, batch-dim, ...
    stride_kbh, stride_kdn, stride_kn,         # K strides: batch-head, dim-N, ...
    stride_mbh, stride_mm,                     # M strides
    stride_obh, stride_om,                     # Out strides
    B, H, N_CTX_Q, N_CTX_K, dim_head,          # batch, heads, context sizes, head dimension
    sm_scale,                                  # scale factor
    sliding_window,                            # bool/int for sliding window
    BLOCK_M: tl.constexpr,                     # block size in M dimension (Q dimension)
    BLOCK_N: tl.constexpr                      # block size in N dimension (K dimension)
):
    # program ids for block-level tiling
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    # derive batch and head from a fused axis if needed (for simplicity, just an example)
    bh = tl.program_id(axis=2)
    b = bh // H
    h = bh % H

    # row indices in [0..N_CTX_Q), col indices in [0..N_CTX_K)
    row_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    col_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # valid checks
    row_mask = row_offsets < N_CTX_Q
    col_mask = col_offsets < N_CTX_K

    # base pointers for Q, K, M, Out
    Q_base = Q_ptr + b * stride_qbh + h * stride_qbd
    K_base = K_ptr + b * stride_kbh + h * stride_kdn
    M_base = M_ptr + b * stride_mbh + h * stride_mm
    O_base = Out_ptr + b * stride_obh + h * stride_om

    # accumulator for partial dot product
    # shape: (BLOCK_M, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over dimension of the head in chunks of 1 (naive)
    # You can increase block loading in the head-dim direction for better performance.
    for d in range(dim_head):
        # load one element from Q and (1) from K across blocks
        q_val = tl.load(
            Q_base + row_offsets * stride_qm + d,
            mask=row_mask,
            other=0.0
        ).to(tl.float32)
        k_val = tl.load(
            K_base + d * stride_kn + col_offsets,
            mask=col_mask,
            other=0.0
        ).to(tl.float32)
        # broadcast them for outer product
        q_val = q_val[:, None]  # (BLOCK_M, 1)
        k_val = k_val[None, :]  # (1, BLOCK_N)
        acc += q_val * k_val

    # scale
    acc = acc * sm_scale

    # optionally apply mask
    # load mask (1 if valid, 0 if invalid) or any broadcastable shape
    # shape expected: (BLOCK_M, BLOCK_N)
    mask_val = tl.load(
        M_base + row_offsets[:, None] * stride_mm + col_offsets[None, :],
        mask=row_mask[:, None] & col_mask[None, :],
        other=1.0  # default to 1.0 (meaning no mask) if out of range
    ).to(tl.float32)

    # sliding window logic can be applied if sliding_window != 0
    # This is a stub showing how you might conditionally override the mask.
    if sliding_window != 0:
        # Example: set some positions to 0 if outside a window
        # This snippet is just a placeholder.
        # row_offsets[:, None], col_offsets[None, :] -> positions
        pass

    # apply mask: if mask == 0, set score to -inf
    neg_inf = float('-inf')
    acc = tl.where(mask_val != 0.0, acc, neg_inf)

    # store final result to output
    tl.store(
        O_base + row_offsets[:, None] * stride_om + col_offsets[None, :],
        acc,
        mask=row_mask[:, None] & col_mask[None, :]
    )


def get_score(Q, K, M, Out, 
              sm_scale, 
              sliding_window=False,
              BLOCK_M=64, 
              BLOCK_N=64):
    """
    Q:  (B, H, N_CTX_Q, dim_head)  float32
    K:  (B, H, dim_head, N_CTX_K)  float32
    M:  (B, H, N_CTX_Q, N_CTX_K)   float32 (mask)
    Out: (B, H, N_CTX_Q, N_CTX_K)  float32
    sm_scale: scale factor for attention
    sliding_window: bool or int for windowed attention
    BLOCK_M, BLOCK_N: tiling dims
    """
    import math
    # shapes
    B, H, N_CTX_Q, dim_head = Q.shape
    _, _, _, N_CTX_K = K.shape

    # strides
    # Q strides
    stride_qbh = Q.stride(0)  # batch-head
    stride_qbd = Q.stride(1)  # head-dim
    stride_qm  = Q.stride(2)  # M dimension
    # K strides
    stride_kbh = K.stride(0)
    stride_kdn = K.stride(1)
    stride_kn  = K.stride(2)
    # M strides
    stride_mbh = M.stride(0)
    stride_mm  = M.stride(2)
    # Out strides
    stride_obh = Out.stride(0)
    stride_om  = Out.stride(2)

    # grid is decided by how many blocks we need in each dimension
    def grid(meta):
        # number of blocks along Q, along K, along B*H
        grid_m = math.ceil(N_CTX_Q / meta['BLOCK_M'])
        grid_n = math.ceil(N_CTX_K / meta['BLOCK_N'])
        grid_bh = B * H
        return (grid_m, grid_n, grid_bh)

    # Try launching kernel with fallback
    while True:
        try:
            _score_kernel[grid](
                Q, K, M, Out,
                stride_qbh, stride_qbd, stride_qm,
                stride_kbh, stride_kdn, stride_kn,
                stride_mbh, stride_mm,
                stride_obh, stride_om,
                B, H, N_CTX_Q, N_CTX_K, dim_head,
                sm_scale,
                int(sliding_window),
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N
            )
            break
        except triton.OutOfResources:
            # reduce tiling dims by half and retry
            BLOCK_M //= 2
            BLOCK_N //= 2
            if BLOCK_M < 1 or BLOCK_N < 1:
                raise RuntimeError("Cannot launch kernel, even with minimal block sizes!")
