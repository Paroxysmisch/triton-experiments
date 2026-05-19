import triton
import triton.language as tl

@triton.jit
def _score_kernel(
    Q, K, M, Out,
    stride_qm, stride_qk, stride_kn, stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    scale
):
    # Matrix multiplication block
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(Out.shape[0], BLOCK_M)
    num_pid_n = tl.cdiv(Out.shape[1], BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    # Offset pointers for Q, K, and Out
    Q = Q + pid_m * BLOCK_M * stride_qm
    K = K + pid_n * BLOCK_N * stride_kn
    Out = Out + pid_m * BLOCK_M * stride_om + pid_n * BLOCK_N * stride_on
    M = M + pid_m * BLOCK_M * stride_om + pid_n * BLOCK_N

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Load mask
    mask = tl.load(M)

    # Main loop
    for k in range(0, Q.shape[1], BLOCK_DMODEL):
        q = tl.load(Q + k * stride_qk, mask=(k + tl.arange(0, BLOCK_DMODEL)) < Q.shape[1], other=0.0)
        k = tl.load(K + k * stride_qk, mask=(k + tl.arange(0, BLOCK_DMODEL)) < K.shape[0], other=0.0)
        acc += tl.dot(q, k)

    # Apply scale and mask
    acc = acc * scale
    acc = tl.where(mask, acc, float('-inf'))

    # Store result
    tl.store(Out, acc, mask=mask)

# Wrapper function
def get_score(q, k, mask, scale):
    BLOCK_M, BLOCK_N, BLOCK_DMODEL = 32, 32, 32
    batch_size, num_heads, seq_len, d_head = q.shape

    # Reshape inputs
    q = q.reshape(batch_size * num_heads, seq_len, d_head)
    k = k.reshape(batch_size * num_heads, seq_len, d_head)
    mask = mask.reshape(batch_size * num_heads, seq_len, seq_len)

    # Prepare output
    output = torch.empty((batch_size * num_heads, seq_len, seq_len), device=q.device, dtype=q.dtype)

    # Configure grid
    grid = (triton.cdiv(seq_len, BLOCK_M) * triton.cdiv(seq_len, BLOCK_N),)

    # Launch kernel
    try:
        _score_kernel[grid](
            q, k, mask, output,
            q.stride(0), q.stride(2), k.stride(1), output.stride(0), output.stride(1),
            BLOCK_M, BLOCK_N, BLOCK_DMODEL,
            scale
        )
    except triton.OutOfResources:
        # If we run out of resources, reduce block sizes and try again
        BLOCK_M, BLOCK_N, BLOCK_DMODEL = 16, 16, 16
        grid = (triton.cdiv(seq_len, BLOCK_M) * triton.cdiv(seq_len, BLOCK_N),)
        _score_kernel[grid](
            q, k, mask, output,
            q.stride(0), q.stride(2), k.stride(1), output.stride(0), output.stride(1),
            BLOCK_M, BLOCK_N, BLOCK_DMODEL,
            scale
        )

    return output.reshape(batch_size, num_heads, seq_len, seq_len)
