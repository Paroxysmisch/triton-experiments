import torch
import triton
import triton.language as tl

@triton.jit
def _score_kernel(
    Q, K, M, Out,
    stride_qm, stride_qk,
    stride_km, stride_kk,
    stride_mm, stride_mn,
    stride_outm, stride_outn,
    sm_scale,
    M_seq: tl.constexpr, N_seq: tl.constexpr, d_model: tl.constexpr,
    window_left: tl.constexpr,
    window_right: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    mask_m = offs_m < M_seq
    mask_n = offs_n < N_seq
    
    # Load Q block
    q = tl.load(
        Q + offs_m[:, None] * stride_qm + tl.arange(0, d_model)[None, :] * stride_qk,
        mask=mask_m[:, None] & (tl.arange(0, d_model)[None, :] < d_model),
        other=0.0
    )
    
    # Load K block
    k = tl.load(
        K + offs_n[:, None] * stride_km + tl.arange(0, d_model)[None, :] * stride_kk,
        mask=mask_n[:, None] & (tl.arange(0, d_model)[None, :] < d_model),
        other=0.0
    )
    
    # Compute scores
    scores = tl.dot(q, tl.trans(k)) * sm_scale
    
    # Apply sliding window mask
    query_idx = offs_m[:, None]
    key_idx = offs_n[None, :]
    window_mask = (key_idx >= (query_idx - window_left)) & (key_idx <= (query_idx + window_right))
    
    # Apply input mask if provided
    if M is not None:
        mask = tl.load(
            M + offs_m[:, None] * stride_mm + offs_n[None, :] * stride_mn,
            mask=mask_m[:, None] & mask_n[None, :],
            other=False
        )
        window_mask &= mask
    
    # Set out-of-window/masked scores to -inf
    scores = tl.where(window_mask, scores, float('-inf'))
    
    # Write output
    out_ptrs = Out + offs_m[:, None] * stride_outm + offs_n[None, :] * stride_outn
    tl.store(out_ptrs, scores, mask=mask_m[:, None] & mask_n[None, :])


def get_score(Q: torch.Tensor, K: torch.Tensor, M: torch.Tensor = None, 
              window_left: int = 0, window_right: int = 0, sm_scale: float = None):
    assert Q.is_cuda and K.is_cuda and (M is None or M.is_cuda)
    assert Q.shape[1] == K.shape[1], "Q and K must have the same feature dimension"
    
    M_seq, d_model = Q.shape
    N_seq, _ = K.shape
    
    sm_scale = sm_scale if sm_scale is not None else 1.0 / (d_model ** 0.5)
    
    Out = torch.empty((M_seq, N_seq), device=Q.device, dtype=Q.dtype)
    
    BLOCK_M, BLOCK_N = 128, 128  # Initial block sizes
    
    while True:
        try:
            grid = (triton.cdiv(M_seq, BLOCK_M), triton.cdiv(N_seq, BLOCK_N))
            
            _score_kernel[grid](
                Q, K, M, Out,
                Q.stride(0), Q.stride(1),
                K.stride(0), K.stride(1),
                M.stride(0) if M is not None else 0,
                M.stride(1) if M is not None else 0,
                Out.stride(0), Out.stride(1),
                sm_scale,
                M_seq, N_seq, d_model,
                window_left, window_right,
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            )
            break
        except triton.OutOfResources:
            # Reduce block sizes and retry
            BLOCK_M //= 2
            BLOCK_N //= 2
            if BLOCK_M < 16 or BLOCK_N < 16:
                raise RuntimeError("Failed to launch kernel with minimum block size")
    
    return Out
