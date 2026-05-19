import triton
import triton.language as tl
import torch

# Define the Triton kernel
@triton.jit
def _fwd_kernel(
    Q, K, V, O,
    sm_scale,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Lk,
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # Program IDs
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    block_m = tl.program_id(2)

    # Offsets for the batch and head
    q_offset = batch_idx * stride_qz + head_idx * stride_qh
    k_offset = batch_idx * stride_kz + head_idx * stride_kh
    v_offset = batch_idx * stride_vz + head_idx * stride_vh
    o_offset = batch_idx * stride_oz + head_idx * stride_oh

    # Initialize accumulators
    m = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    # Loop over blocks of K and V
    for block_n in range(0, Lk, BLOCK_N):
        # Load Q, K, V
        q = tl.load(Q + q_offset + (block_m * BLOCK_M + tl.arange(0, BLOCK_M))[:, None] * stride_qm + tl.arange(0, BLOCK_N) * stride_qk)
        k = tl.load(K + k_offset + (block_n + tl.arange(0, BLOCK_N))[:, None] * stride_kn + tl.arange(0, BLOCK_N) * stride_kk)
        v = tl.load(V + v_offset + (block_n + tl.arange(0, BLOCK_N))[:, None] * stride_vn + tl.arange(0, BLOCK_N) * stride_vk)

        # Compute QK^T
        qk = tl.dot(q, k.T) * sm_scale

        # Apply causal mask
        if IS_CAUSAL:
            mask = (block_m * BLOCK_M + tl.arange(0, BLOCK_M))[:, None] >= (block_n + tl.arange(0, BLOCK_N))
            qk = tl.where(mask, qk, float('-inf'))

        # Compute max for numerical stability
        max_qk = tl.max(qk, 1)
        m = tl.maximum(m, max_qk)

        # Compute exp and accumulate
        exp_qk = tl.exp2(qk - max_qk[:, None])
        acc += tl.dot(exp_qk, v)

        # Accumulate l
        l += tl.sum(exp_qk, 1)

    # Normalize and write output
    o = acc / l[:, None]
    tl.store(O + o_offset + (block_m * BLOCK_M + tl.arange(0, BLOCK_M))[:, None] * stride_om + tl.arange(0, BLOCK_N) * stride_ok, o)

# Python wrapper function
def flash_attn_triton(q, k, v, sm_scale, is_causal=False):
    batch_size, heads, seq_len, dims = q.shape
    BLOCK_M = 128  # or another optimal block size
    BLOCK_N = 128  # or another optimal block size

    # Allocate output tensor
    o = torch.empty_like(q)

    # Define grid dimensions
    grid = (batch_size, heads, seq_len // BLOCK_M)

    # Launch Triton kernel
    num_warps = 4 if seq_len <= 64 else 8  # Adjust based on seq_len for optimal performance
    _fwd_kernel[grid](
        q, k, v, o,
        sm_scale,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        seq_len,
        IS_CAUSAL=is_causal,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=num_warps
    )

    return o
