import torch
import triton
import triton.language as tl

@triton.jit
def _fused_bmm_rmsnorm_gelu_dropout_kernel(
    input1_ptr, input2_ptr, output_ptr,
    B, N, M, P,
    stride_b1, stride_n1, stride_m1,
    stride_b2, stride_m2, stride_p2,
    stride_bout, stride_nout, stride_pout,
    eps, dropout_p, is_training,
    APPROXIMATE_TANH: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_p_blocks = tl.cdiv(P, BLOCK_P)
    b = pid // (N * num_p_blocks)
    n = (pid // num_p_blocks) % N
    p_block = pid % num_p_blocks

    p_start = p_block * BLOCK_P
    p_offsets = p_start + tl.arange(0, BLOCK_P)
    p_mask = p_offsets < P

    # Accumulate BMM results
    acc = tl.zeros((BLOCK_P,), dtype=tl.float32)
    for m_block in range(0, tl.cdiv(M, BLOCK_M)):
        m_start = m_block * BLOCK_M
        m_offsets = m_start + tl.arange(0, BLOCK_M)
        m_mask = m_offsets < M

        a_ptr = input1_ptr + b * stride_b1 + n * stride_n1 + m_offsets * stride_m1
        a = tl.load(a_ptr, mask=m_mask, other=0.0)

        b_ptr = input2_ptr + b * stride_b2 + m_offsets[:, None] * stride_m2 + p_offsets[None, :] * stride_p2
        b = tl.load(b_ptr, mask=m_mask[:, None] & p_mask[None, :], other=0.0)

        acc += tl.sum(a[:, None] * b, axis=0)

    # Compute RMS normalization
    sum_squares = tl.sum(acc * acc, axis=0)
    mean_squares = sum_squares / P
    rms = tl.sqrt(mean_squares + eps)
    normalized = acc / rms

    # GELU activation
    if APPROXIMATE_TANH:
        gelu = 0.5 * normalized * (1 + tl.tanh(tl.sqrt(2 / tl.math.pi) * (normalized + 0.044715 * normalized ** 3)))
    else:
        gelu = 0.5 * normalized * (1 + tl.erf(normalized / tl.sqrt(2.0)))

    # Dropout
    if is_training:
        random = tl.rand(tl.rand_seed, p_offsets)
        keep_mask = random > dropout_p
        gelu = tl.where(keep_mask, gelu / (1 - dropout_p), 0.0)

    # Write output
    out_ptr = output_ptr + b * stride_bout + n * stride_nout + p_offsets * stride_pout
    tl.store(out_ptr, gelu, mask=p_mask)

def fused_bmm_rmsnorm_gelu_dropout(
    input1: torch.Tensor, input2: torch.Tensor, normalized_shape,
    dropout_p: float = 0.1, eps: float = 1e-5, training: bool = True,
    approximate: str = 'none', out: torch.Tensor = None
) -> torch.Tensor:
    B, N, M = input1.shape
    B2, M2, P = input2.shape
    assert B == B2 and M == M2, "Input shapes incompatible for BMM"
    assert tuple(normalized_shape) == (P,), "RMSNorm normalized_shape must be (P)"
    output = torch.empty((B, N, P), device=input1.device, dtype=input1.dtype) if out is None else out

    BLOCK_P = 128
    BLOCK_M = 32
    grid = (B * N * triton.cdiv(P, BLOCK_P),)
    approx = approximate == 'tanh'

    _fused_bmm_rmsnorm_gelu_dropout_kernel[grid](
        input1, input2, output,
        B, N, M, P,
        input1.stride(0), input1.stride(1), input1.stride(2),
        input2.stride(0), input2.stride(1), input2.stride(2),
        output.stride(0), output.stride(1), output.stride(2),
        eps, dropout_p, training,
        APPROXIMATE_TANH=approx,
        BLOCK_P=BLOCK_P,
        BLOCK_M=BLOCK_M,
    )
    return output
