import torch
import triton
import triton.language as tl


@triton.jit
def _score_kernel(
    Q,         # [n_ctx_q, d_model]
    K,         # [n_ctx_k, d_model]
    M,         # [n_ctx_q, n_ctx_k] or None
    Out,       # [n_ctx_q, n_ctx_k]
    n_ctx_q: tl.constexpr,
    n_ctx_k: tl.constexpr,
    d_model: tl.constexpr,
    stride_q_ctx: tl.constexpr,
    stride_k_ctx: tl.constexpr,
    stride_m_ctx: tl.constexpr,
    stride_out_ctx: tl.constexpr,
    sm_scale: float,
    sliding_size: int,   # 0 means no sliding window
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Compute scaled attention scores = (Q * K^T) * sm_scale, with optional mask M.
    If sliding_size > 0, each row i only interacts with columns [i-sliding_size, i+sliding_size].
    """
    pid = tl.program_id(0)
    grid_n = (n_ctx_k + BLOCK_N - 1) // BLOCK_N
    pid_m = pid // grid_n
    pid_n = pid % grid_n

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Bounds check
    rm_mask = rm < n_ctx_q
    rn_mask = rn < n_ctx_k

    # If sliding_size > 0, compute sliding bounds
    if sliding_size > 0:
        left_bound = rm - sliding_size
        right_bound = rm + sliding_size
        valid_col = (rn[None, :] >= left_bound[:, None]) & (rn[None, :] <= right_bound[:, None])
    else:
        valid_col = True

    # Initialize an accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Blocked dot product
    rd = tl.arange(0, BLOCK_D)
    # Pointers for Q, K
    q_ptrs = Q + (rm[:, None] * stride_q_ctx + rd[None, :] * 1)
    k_ptrs = K + (rn[None, :] * stride_k_ctx + rd[:, None] * 1)

    # d_model may not be multiple of BLOCK_D
    d_left = d_model
    while d_left > 0:
        d_chunk = tl.where(d_left > BLOCK_D, BLOCK_D, d_left)
        # Load
        q_vals = tl.load(q_ptrs, mask=(rm_mask[:, None] & (rd[None, :] < d_chunk)), other=0.0)
        k_vals = tl.load(k_ptrs, mask=(rn_mask[None, :] & (rd[:, None] < d_chunk)), other=0.0)
        # Dot
        acc += tl.dot(q_vals, k_vals)
        # Advance pointers
        q_ptrs += BLOCK_D
        k_ptrs += BLOCK_D
        d_left -= d_chunk

    # Scale
    acc = acc * sm_scale

    # Apply mask if provided
    if M is not None:
        m_ptrs = M + (rm[:, None] * stride_m_ctx + rn[None, :])
        mask_vals = tl.load(m_ptrs, mask=(rm_mask[:, None] & rn_mask[None, :]), other=0.0)
        # Typically, mask is a boolean or 0/1, but can be large negative for OOM attention as well
        # Here, assume 1.0 => keep, 0.0 => discard for simplicity:
        acc = tl.where(mask_vals > 0, acc, float("-inf"))

    # If sliding_size > 0, enforce that beyond the window we set -inf
    if sliding_size > 0:
        acc = tl.where(valid_col, acc, float("-inf"))

    # Store
    out_ptrs = Out + (rm[:, None] * stride_out_ctx + rn[None, :])
    tl.store(out_ptrs, acc, mask=(rm_mask[:, None] & rn_mask[None, :]))


def get_score(Q, K, M=None, sm_scale=0.125, sliding_size=0,
              BLOCK_M=64, BLOCK_N=64, BLOCK_D=32):
    """
    Python wrapper to call _score_kernel.
    Q: [n_ctx_q, d_model]
    K: [n_ctx_k, d_model]
    M: [n_ctx_q, n_ctx_k] or None
    sm_scale: scaling factor for QK^T
    sliding_size: if > 0, use sliding window attention
    BLOCK_M, BLOCK_N, BLOCK_D: initial block sizes
    """
    device = Q.device
    n_ctx_q, d_model = Q.shape
    n_ctx_k, d_model_k = K.shape
    assert d_model == d_model_k, "Q/K shape mismatch"

    # If M is given, check shape
    if M is not None:
        assert M.shape[0] == n_ctx_q and M.shape[1] == n_ctx_k, "Mask shape mismatch"
    # Prepare output
    Out = torch.empty((n_ctx_q, n_ctx_k), dtype=Q.dtype, device=device)

    # Strides for kernel
    stride_q_ctx = Q.stride(0)
    stride_k_ctx = K.stride(0)
    stride_out_ctx = Out.stride(0)
    stride_m_ctx = M.stride(0) if M is not None else 0

    # Helpers
    def grid(meta):
        return (
            triton.cdiv(n_ctx_q, meta["BLOCK_M"]) *
            triton.cdiv(n_ctx_k, meta["BLOCK_N"]),
        )

    # Retry with smaller blocks if resource constraints occur
    while True:
        try:
            _score_kernel[grid](
                Q, K, M, Out,
                n_ctx_q, n_ctx_k, d_model,
                stride_q_ctx, stride_k_ctx, stride_m_ctx, stride_out_ctx,
                sm_scale, sliding_size,
                BLOCK_M, BLOCK_N, BLOCK_D,
            )
            break
        except triton.cuda_backend.CudaOutOfResourcesError:
            # Reduce block sizes by half and retry
            BLOCK_M = max(BLOCK_M // 2, 16)
            BLOCK_N = max(BLOCK_N // 2, 16)
            BLOCK_D = max(BLOCK_D // 2, 16)
            if BLOCK_M < 16 or BLOCK_N < 16 or BLOCK_D < 16:
                raise RuntimeError("Could not launch kernel with given dimensions.")

    return Out
