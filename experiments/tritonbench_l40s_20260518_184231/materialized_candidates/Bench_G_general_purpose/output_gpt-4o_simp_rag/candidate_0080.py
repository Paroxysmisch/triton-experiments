import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_aligned(
    Q, K, V, B0, Out,
    stride_qbs, stride_qh, stride_kbs, stride_kh, stride_vbs, stride_vh, stride_obs, stride_oh,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr
):
    # Obtain program IDs for grid dimensions
    batch_id = tl.program_id(0)
    head_id = tl.program_id(1)
    start_m = tl.program_id(2)

    # Calculate offsets and indices
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)

    # Calculate pointers for Q, K, V
    q_ptrs = Q + (batch_id * stride_qbs + head_id * stride_qh + offs_m[:, None] * stride_qbs + offs_d[None, :])
    k_ptrs = K + (batch_id * stride_kbs + head_id * stride_kh + offs_n[None, :] * stride_kbs + offs_d[:, None])
    v_ptrs = V + (batch_id * stride_vbs + head_id * stride_vh + offs_n[:, None] * stride_vbs + offs_d[None, :])

    # Load Q
    q = tl.load(q_ptrs, mask=offs_m[:, None] < Q.shape[2], other=0.0)

    # Initialize accumulators
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # Loop over K and V
    for start_n in range(0, K.shape[2], BLOCK_N):
        # Load K and V
        k = tl.load(k_ptrs + start_n * stride_kbs, mask=(start_n + offs_n[None, :]) < K.shape[2], other=0.0)
        v = tl.load(v_ptrs + start_n * stride_vbs, mask=(start_n + offs_n[:, None]) < V.shape[2], other=0.0)

        # Compute QK^T
        qk = tl.dot(q, k)

        # Apply bias
        b_ptrs = B0 + (batch_id * B0.stride(0) + head_id * B0.stride(1) + offs_m[:, None] * B0.stride(2) + (start_n + offs_n[None, :]) * B0.stride(3))
        bias = tl.load(b_ptrs, mask=(start_n + offs_n[None, :]) < B0.shape[3], other=0.0)
        qk += bias

        # Compute softmax
        m = tl.max(qk, axis=1)
        p = tl.math.exp2(qk - m[:, None])
        p = p / tl.sum(p, axis=1)[:, None]

        # Update accumulator
        acc += tl.dot(p, v)

    # Store result
    out_ptrs = Out + (batch_id * stride_obs + head_id * stride_oh + offs_m[:, None] * stride_obs + offs_d[None, :])
    tl.store(out_ptrs, acc, mask=offs_m[:, None] < Out.shape[2])


def _attention_rel_h_rel_w_kernel_aligned_device(q, k, v, b0, out):
    # Define block sizes
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_DMODEL = q.shape[-1]

    # Validate input shapes and types
    assert q.shape[2] == k.shape[2] == v.shape[2]
    assert q.dtype == k.dtype == v.dtype == b0.dtype == out.dtype

    # Calculate grid dimensions
    grid = (q.shape[0], q.shape[1], triton.cdiv(q.shape[2], BLOCK_M))

    # Launch the Triton kernel
    _fwd_kernel_aligned[grid](
        q, k, v, b0, out,
        q.stride(0), q.stride(1), k.stride(0), k.stride(1), v.stride(0), v.stride(1), out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=BLOCK_DMODEL
    )
