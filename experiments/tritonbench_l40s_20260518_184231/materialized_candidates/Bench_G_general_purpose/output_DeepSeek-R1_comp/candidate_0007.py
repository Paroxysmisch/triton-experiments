import torch
import triton
import triton.language as tl

@triton.jit
def ff_llama_kernel(
    x_ptr, w1_ptr, w3_ptr, rms_w_ptr, output_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_w1k, stride_w1n,
    stride_w3k, stride_w3n,
    stride_rms_m,
    stride_om, stride_on,
    EPS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    USE_FP8: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    m_offset = pid_m * BLOCK_SIZE_M
    m_indices = m_offset + tl.arange(0, BLOCK_SIZE_M)
    m_mask = m_indices < M

    n_offset = pid_n * BLOCK_SIZE_N
    n_indices = n_offset + tl.arange(0, BLOCK_SIZE_N)
    n_mask = n_indices < N

    # Load RMS weights for current M block
    rms_offsets = m_offset + tl.arange(0, BLOCK_SIZE_M)
    rms = tl.load(rms_w_ptr + rms_offsets, mask=m_mask, other=0.0)

    # Phase 1: Compute sum of squares for RMS normalization
    sum_sq = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    for k_block in range(0, K, BLOCK_SIZE_K):
        k_indices = k_block + tl.arange(0, BLOCK_SIZE_K)
        k_mask = k_indices < K

        x_offsets = m_indices[:, None] * stride_xm + k_indices[None, :] * stride_xk
        x = tl.load(x_ptr + x_offsets, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        x_sq = x * x
        sum_sq += tl.sum(x_sq, axis=1)

    # Compute RMS scaling factors
    rms_scale = rms / tl.sqrt(sum_sq / K + EPS)

    # Phase 2: Perform scaled matrix multiplications
    acc1 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k_block in range(0, K, BLOCK_SIZE_K):
        k_indices = k_block + tl.arange(0, BLOCK_SIZE_K)
        k_mask = k_indices < K

        # Load and scale input data
        x_offsets = m_indices[:, None] * stride_xm + k_indices[None, :] * stride_xk
        x = tl.load(x_ptr + x_offsets, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        x_scaled = x * rms_scale[:, None]

        # Load weight tiles with appropriate casting
        w1_offsets = k_indices[:, None] * stride_w1k + n_indices[None, :] * stride_w1n
        w1 = tl.load(w1_ptr + w1_offsets, mask=k_mask[:, None] & n_mask[None, :], other=0.0)
        w3_offsets = k_indices[:, None] * stride_w3k + n_indices[None, :] * stride_w3n
        w3 = tl.load(w3_ptr + w3_offsets, mask=k_mask[:, None] & n_mask[None, :], other=0.0)

        if USE_FP8:
            w1 = w1.to(tl.float8e5, bitcast=True).to(tl.float32)
            w3 = w3.to(tl.float8e5, bitcast=True).to(tl.float32)

        # Accumulate matrix products
        acc1 += tl.dot(x_scaled, w1, allow_tf32=True)
        acc2 += tl.dot(x_scaled, w3, allow_tf32=True)

    # Apply fused activation and store result
    silu = acc1 * tl.sigmoid(acc1)
    output = silu * acc2

    # Write output
    out_offsets = m_indices[:, None] * stride_om + n_indices[None, :] * stride_on
    tl.store(output_ptr + out_offsets, output, mask=m_mask[:, None] & n_mask[None, :])

def ff_llama(
    x: torch.Tensor,
    w1: torch.Tensor,
    w3: torch.Tensor,
    rms_w: torch.Tensor,
    EPS: float = 1e-6,
    BLOCK_SIZE_M: int = 64,
    BLOCK_SIZE_N: int = 64,
    BLOCK_SIZE_K: int = 64,
):
    assert x.ndim == 2, "Input x must be 2D"
    M, K = x.shape
    N, K_w = w1.shape
    
    assert K == K_w, "Weight dimensions mismatch"
    assert w3.shape == (N, K), "w3 must match w1 dimensions"
    assert rms_w.shape == (M,), "RMS weights must match input dimension"

    # Transpose weights for efficient access
    w1_t = w1.t().contiguous()
    w3_t = w3.t().contiguous()

    output = torch.empty((M, N), device=x.device, dtype=x.dtype)

    # Determine if using FP8
    USE_FP8 = w1.dtype in (torch.float8_e5m2, torch.float8_e4m3fn)

    def grid(META):
        return (
            triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
            1, 1
        )

    ff_llama_kernel[grid](
        x, w1_t, w3_t, rms_w, output,
        M, N, K,
        x.stride(0), x.stride(1),
        w1_t.stride(0), w1_t.stride(1),
        w3_t.stride(0), w3_t.stride(1),
        rms_w.stride(0),
        output.stride(0), output.stride(1),
        EPS=EPS,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        USE_FP8=USE_FP8,
    )
    return output
