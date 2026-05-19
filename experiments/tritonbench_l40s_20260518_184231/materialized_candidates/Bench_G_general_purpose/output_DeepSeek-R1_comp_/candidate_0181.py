import torch
import triton
import triton.language as tl

@triton.jit
def rms_matmul_rbe(
    # Pointers to matrices
    x_ptr, w_ptr, rms_w_ptr, out_ptr,
    # Matrix dimensions
    M, N, K,
    # Strides
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_rms_wk,
    stride_outm, stride_outn,
    # Other parameters
    start_token_position: tl.constexpr,
    USE_FP8: tl.constexpr,
    RBE_EPILOGUE: tl.constexpr,
    THETA: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    m_offset = pid_m * BLOCK_SIZE_M
    n_offset = pid_n * BLOCK_SIZE_N
    m_mask = (m_offset + tl.arange(0, BLOCK_SIZE_M)) < M
    n_mask = (n_offset + tl.arange(0, BLOCK_SIZE_N)) < N

    # Accumulate sum of squares for RMS
    sum_sq = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    for k_block in range(0, K, BLOCK_SIZE_K):
        k_offsets = k_block + tl.arange(0, BLOCK_SIZE_K)
        k_mask = k_offsets < K

        x_ptr_mask = m_mask[:, None] & k_mask[None, :]
        x_vals = tl.load(x_ptr + m_offset[:, None] * stride_xm + k_offsets[None, :] * stride_xk,
                        mask=x_ptr_mask, other=0.0)
        if USE_FP8:
            x_vals = x_vals.to(tl.float8e5, bitcast=True).to(tl.float32)
        sum_sq += tl.sum(x_vals * x_vals, axis=1)

    rms = tl.math.rsqrt(sum_sq / K + EPS)

    # Matrix multiplication with RMS normalization
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k_block in range(0, K, BLOCK_SIZE_K):
        k_offsets = k_block + tl.arange(0, BLOCK_SIZE_K)
        k_mask = k_offsets < K

        # Load and normalize x
        x_vals = tl.load(x_ptr + m_offset[:, None] * stride_xm + k_offsets[None, :] * stride_xk,
                        mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        rms_w = tl.load(rms_w_ptr + k_offsets * stride_rms_wk,
                        mask=k_mask, other=0.0)
        x_norm = x_vals * rms[:, None] * rms_w[None, :]

        # Load weights
        w_vals = tl.load(w_ptr + k_offsets[:, None] * stride_wk + n_offset[None, :] * stride_wn,
                        mask=k_mask[:, None] & n_mask[None, :], other=0.0)
        if USE_FP8:
            w_vals = w_vals.to(tl.float8e5, bitcast=True).to(tl.float32)

        acc += tl.dot(x_norm.to(tl.float32), w_vals.to(tl.float32))

    # Apply rotary embeddings (simplified example)
    if RBE_EPILOGUE:
        pos = start_token_position + m_offset + tl.arange(0, BLOCK_SIZE_M)[:, None]
        dim = tl.arange(0, BLOCK_SIZE_N)[None, :] // 2
        freq = 1.0 / (THETA ** (2 * dim / BLOCK_SIZE_N))
        angle = pos * freq
        cos = tl.cos(angle)
        sin = tl.sin(angle)
        
        acc0 = acc[:, 0::2] * cos - acc[:, 1::2] * sin
        acc1 = acc[:, 0::2] * sin + acc[:, 1::2] * cos
        acc = tl.view(acc, (BLOCK_SIZE_M, BLOCK_SIZE_N//2, 2))
        acc = tl.cat((acc0[..., None], acc1[..., None]), axis=2)
        acc = tl.view(acc, (BLOCK_SIZE_M, BLOCK_SIZE_N))

    # Write back results
    out_ptrs = out_ptr + (m_offset[:, None] * stride_outm + n_offset[None, :] * stride_outn)
    tl.store(out_ptrs, acc.to(tl.float16), mask=m_mask[:, None] & n_mask[None, :])

def rms_matmul_rbe_qkv_wrapper(
    x: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    rms_w_q: torch.Tensor,
    rms_w_k: torch.Tensor,
    rms_w_v: torch.Tensor,
    start_token_position: int,
    USE_FP8: bool,
    THETA: float = 10000.0,
    EPS: float = 1e-6,
    BLOCK_SIZE_M: int = 64,
    BLOCK_SIZE_N: int = 64,
    BLOCK_SIZE_K: int = 64,
):
    assert x.dim() == 2, "Input x must be 2D"
    M, K = x.shape
    N = w_q.size(-1)
    
    q = torch.empty((M, N), device=x.device, dtype=x.dtype)
    k = torch.empty((M, N), device=x.device, dtype=x.dtype)
    v = torch.empty((M, N), device=x.device, dtype=x.dtype)

    def grid(META): return (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
    
    for out, w, rms_w, rbe_flag in [(q, w_q, rms_w_q, True),
                                   (k, w_k, rms_w_k, True),
                                   (v, w_v, rms_w_v, False)]:
        rms_matmul_rbe[grid](
            x, w, rms_w, out,
            M, N, K,
            x.stride(0), x.stride(1),
            w.stride(0), w.stride(1),
            rms_w.stride(0),
            out.stride(0), out.stride(1),
            start_token_position,
            USE_FP8,
            rbe_flag,
            THETA,
            EPS,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            BLOCK_SIZE_K,
        )
    return q, k, v
