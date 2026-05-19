import torch
import triton
import triton.language as tl
from triton.common import cdiv

@triton.jit
def _score_kernel(
    Q_ptr, K_ptr, M_ptr, Out_ptr,
    n_ctx_q, n_ctx_k, d_model,
    stride_q_ctx, stride_k_ctx,
    stride_q_d, stride_k_d,
    stride_m_q, stride_m_k,
    stride_out_q, stride_out_k,
    sm_scale: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_n = cdiv(n_ctx_k, BLOCK_N)
    pid_m = pid // grid_n
    pid_n = pid % grid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < n_ctx_q
    mask_n = offs_n < n_ctx_k

    Q_block_ptr = tl.make_block_ptr(
        base=Q_ptr,
        shape=(n_ctx_q, d_model),
        strides=(stride_q_ctx, stride_q_d),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_D),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        base=K_ptr,
        shape=(d_model, n_ctx_k),
        strides=(stride_k_d, stride_k_ctx),
        offsets=(0, pid_n * BLOCK_N),
        block_shape=(BLOCK_D, BLOCK_N),
        order=(0, 1),
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for d in range(0, d_model, BLOCK_D):
        q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
        k = tl.load(K_block_ptr, boundary_check=(1, 0), padding_option="zero")
        acc += tl.dot(q, k, allow_tf32=True)
        Q_block_ptr = tl.advance(Q_block_ptr, (0, BLOCK_D))
        K_block_ptr = tl.advance(K_block_ptr, (BLOCK_D, 0))

    acc = acc * sm_scale

    m_offs = offs_m[:, None] * stride_m_q + offs_n[None, :] * stride_m_k
    mask_val = tl.load(M_ptr + m_offs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
    acc += mask_val

    out_offs = offs_m[:, None] * stride_out_q + offs_n[None, :] * stride_out_k
    tl.store(Out_ptr + out_offs, acc, mask=mask_m[:, None] & mask_n[None, :])

def get_score(Q, K, M):
    assert Q.dim() == 2 and K.dim() == 2 and M.dim() == 2, "Inputs must be 2D tensors"
    n_ctx_q, d_model = Q.shape
    n_ctx_k = K.shape[0]
    assert K.shape[1] == d_model, "Feature dimension mismatch"
    assert M.shape == (n_ctx_q, n_ctx_k), "Mask shape mismatch"

    Q = Q if Q.is_contiguous() else Q.contiguous()
    K = K if K.is_contiguous() else K.contiguous()
    M = M if M.is_contiguous() else M.contiguous()

    Out = torch.empty((n_ctx_q, n_ctx_k), device=Q.device, dtype=Q.dtype)
    sm_scale = 1.0 / (d_model ** 0.5)

    BLOCK_M, BLOCK_N, BLOCK_D = 64, 64, 32
    max_retries = 5

    for retry in range(max_retries + 1):
        try:
            def grid(meta):
                return (cdiv(n_ctx_q, meta['BLOCK_M']) * cdiv(n_ctx_k, meta['BLOCK_N']),)

            _score_kernel[grid](
                Q, K, M, Out,
                n_ctx_q, n_ctx_k, d_model,
                Q.stride(0), K.stride(0),
                Q.stride(1), K.stride(1),
                M.stride(0), M.stride(1),
                Out.stride(0), Out.stride(1),
                sm_scale=sm_scale,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                BLOCK_D=BLOCK_D,
            )
            return Out
        except triton.OutOfResources:
            if retry < max_retries:
                BLOCK_M //= 2
                BLOCK_N //= 2
                BLOCK_D = max(BLOCK_D // 2, 16)
                print(f"Retrying with reduced block sizes: BLOCK_M={BLOCK_M}, BLOCK_N={BLOCK_N}, BLOCK_D={BLOCK_D}")
            else:
                raise RuntimeError("Maximum retries exceeded. Further reducing block sizes may not be feasible.")
    return Out
