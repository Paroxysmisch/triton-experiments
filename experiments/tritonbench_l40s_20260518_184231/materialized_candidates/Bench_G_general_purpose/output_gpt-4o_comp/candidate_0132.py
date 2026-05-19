import triton
import triton.language as tl
import torch

# Define constants for block sizes
BLOCK_M = 128
BLOCK_N = 128
BLOCK_DMODEL = 64

@triton.jit
def _fwd_kernel(Q, K, V, sm_scale, Out, m_size, n_size, d_size, head_size, batch, USE_FP8, IS_CAUSAL, stride_qm, stride_kn, stride_vn, stride_outm, stride_outn, **meta):
    pid = tl.program_id(0)
    head = tl.program_id(1)

    # Compute the offsets for Q, K, V
    q_offset = pid * BLOCK_M
    k_offset = 0
    v_offset = 0

    # Load Q
    Q_block = tl.load(Q + q_offset * stride_qm + head * d_size, mask=q_offset + tl.arange(0, BLOCK_M) < m_size)

    # Initialize accumulators
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # Loop over the K and V matrices
    for nk in range(0, n_size, BLOCK_N):
        # Load K and V
        K_block = tl.load(K + k_offset * stride_kn + head * d_size, mask=k_offset + tl.arange(0, BLOCK_N) < n_size)
        V_block = tl.load(V + v_offset * stride_vn + head * d_size, mask=v_offset + tl.arange(0, BLOCK_N) < n_size)

        # Compute QK^T
        logits = tl.dot(Q_block, tl.trans(K_block)) * sm_scale

        # Apply causal mask if needed
        if IS_CAUSAL:
            mask = tl.arange(0, BLOCK_M)[:, None] >= (nk + tl.arange(0, BLOCK_N))[None, :]
            logits = tl.where(mask, float('-inf'), logits)

        # Compute softmax
        max_logits = tl.max(logits, axis=1)
        logits = logits - max_logits[:, None]
        exp_logits = tl.exp(logits)
        sum_exp_logits = tl.sum(exp_logits, axis=1)
        softmax = exp_logits / sum_exp_logits[:, None]

        # Update accumulator with weighted values
        acc += tl.dot(softmax, V_block)

        # Update offsets
        k_offset += BLOCK_N
        v_offset += BLOCK_N

    # Store the result
    tl.store(Out + q_offset * stride_outm + head * d_size, acc, mask=q_offset + tl.arange(0, BLOCK_M) < m_size)


def triton_fa(Q, K, V, sm_scale, USE_FP8=False, IS_CAUSAL=False):
    # Check data types and convert if necessary
    assert Q.dtype == K.dtype == V.dtype, "Q, K, V must have the same dtype"
    dtype = Q.dtype

    # Get dimensions
    batch, head_size, m_size, d_size = Q.shape
    _, _, n_size, _ = K.shape

    # Prepare output tensor
    Out = torch.empty((batch, head_size, m_size, d_size), dtype=dtype, device=Q.device)

    # Compute grid size
    grid = (triton.cdiv(m_size, BLOCK_M), head_size * batch)

    # Launch kernel
    _fwd_kernel[grid](
        Q, K, V, sm_scale, Out,
        m_size, n_size, d_size, head_size, batch,
        USE_FP8, IS_CAUSAL,
        Q.stride(2), K.stride(2), V.stride(2), Out.stride(2), Out.stride(3),
        num_warps=4 if BLOCK_N <= 128 else 8,
        num_stages=2
    )

    return Out
