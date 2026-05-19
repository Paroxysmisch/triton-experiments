import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_aligned(
    Q, K, V, B0, Out, sm_scale,
    stride_qbs, stride_qh, stride_kbs, stride_kh, stride_vbs, stride_vh,
    stride_obs, stride_oh, stride_bbs, stride_bh,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_m = tl.program_id(2)

    # Offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    q = tl.load(Q + (cur_batch * stride_qbs + cur_head * stride_qh + offs_m[:, None] * stride_qbs + offs_d[None, :]), mask=offs_m[:, None] < Q.shape[0], other=0.0)
    k = tl.load(K + (cur_batch * stride_kbs + cur_head * stride_kh + offs_n[None, :] * stride_kbs + offs_d[:, None]), mask=offs_n[None, :] < K.shape[0], other=0.0)
    v = tl.load(V + (cur_batch * stride_vbs + cur_head * stride_vh + offs_n[:, None] * stride_vbs + offs_d[None, :]), mask=offs_n[:, None] < V.shape[0], other=0.0)

    # Compute QK^T
    qk = tl.dot(q, k, trans_b=True)
    qk *= sm_scale

    # Apply bias
    b = tl.load(B0 + (cur_batch * stride_bbs + cur_head * stride_bh + offs_m[:, None] * stride_bbs + offs_n[None, :]), mask=offs_m[:, None] < B0.shape[0], other=0.0)
    qk += b

    # Softmax
    m = tl.max(qk, axis=1)
    qk = tl.math.exp2(qk - m[:, None])
    l = tl.sum(qk, axis=1)
    qk /= l[:, None]

    # Compute weighted sum of V
    o = tl.dot(qk, v)

    # Store result
    tl.store(Out + (cur_batch * stride_obs + cur_head * stride_oh + offs_m[:, None] * stride_obs + offs_d[None, :]), o, mask=offs_m[:, None] < Out.shape[0])


def _attention_rel_h_rel_w_kernel_aligned_device(q, k, v, b0, out, sm_scale, num_warps, num_stages):
    # Ensure shapes and types are consistent
    assert q.shape[0] == k.shape[0] == v.shape[0] == b0.shape[0] == out.shape[0]  # Batch size
    assert q.shape[1] == k.shape[1] == v.shape[1] == b0.shape[1] == out.shape[1]  # Head dimension
    assert q.shape[2] == out.shape[2]  # Sequence length
    assert q.shape[3] == k.shape[3] == v.shape[3] == b0.shape[3] == out.shape[3]  # Model dimension

    # Define grid
    batch_size, num_heads, seq_len, d_model = q.shape
    grid = (batch_size, num_heads, triton.cdiv(seq_len, BLOCK_M))

    # Launch kernel
    _fwd_kernel_aligned[grid](
        q, k, v, b0, out, sm_scale,
        q.stride(0), q.stride(1), k.stride(0), k.stride(1), v.stride(0), v.stride(1),
        out.stride(0), out.stride(1), b0.stride(0), b0.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=d_model,
        num_warps=num_warps, num_stages=num_stages
    )

# Constants for BLOCK sizes
BLOCK_M = 128
BLOCK_N = 128
BLOCK_DMODEL = 64

# Example usage
# Assuming q, k, v, b0, out are already defined tensors with appropriate shapes and types
_attention_rel_h_rel_w_kernel_aligned_device(q, k, v, b0, out, sm_scale=1.0, num_warps=4, num_stages=1)
