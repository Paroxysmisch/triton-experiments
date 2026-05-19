import triton
import triton.language as tl

@triton.jit
def _attn_fwd(
    Q, K, V, O, 
    q_scale, k_scale, 
    stride_qm, stride_qk, 
    stride_km, stride_kk, 
    stride_vm, stride_vk, 
    stride_om, stride_ok, 
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, 
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_warp = BLOCK_M * BLOCK_N // tl.warp_size()
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    block_offset_m = pid_m * BLOCK_M
    block_offset_n = pid_n * BLOCK_N

    # Pointers for Q, K, V, O
    Q_block_ptr = tl.make_block_ptr(
        base=Q, shape=(M, K), strides=(stride_qm, stride_qk),
        offsets=(block_offset_m, 0), block_shape=(BLOCK_M, BLOCK_K), order=(1, 0)
    )
    K_block_ptr = tl.make_block_ptr(
        base=K, shape=(K, N), strides=(stride_km, stride_kk),
        offsets=(0, block_offset_n), block_shape=(BLOCK_K, BLOCK_N), order=(0, 1)
    )
    V_block_ptr = tl.make_block_ptr(
        base=V, shape=(K, N), strides=(stride_vm, stride_vk),
        offsets=(0, block_offset_n), block_shape=(BLOCK_K, BLOCK_N), order=(0, 1)
    )
    O_block_ptr = tl.make_block_ptr(
        base=O, shape=(M, N), strides=(stride_om, stride_ok),
        offsets=(block_offset_m, block_offset_n), block_shape=(BLOCK_M, BLOCK_N), order=(1, 0)
    )

    # Initialize accumulators
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M, 1), dtype=tl.float32)
    m_i = tl.full((BLOCK_M, 1), float("-inf"), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        q = tl.load(Q_block_ptr)
        k = tl.load(K_block_ptr)
        v = tl.load(V_block_ptr)

        # Compute dot products and scale
        qk = tl.dot(q, k, allow_tf32=True) * q_scale * k_scale

        # Compute max for numerical stability
        m_i_new = tl.maximum(m_i, tl.max(qk, 1, keepdim=True))
        qk = qk - m_i_new
        qk = tl.exp(qk)

        # Compute sum for normalization
        l_i_new = l_i * tl.exp(m_i - m_i_new) + tl.sum(qk, 1, keepdim=True)
        qk = qk / l_i_new

        # Accumulate weighted sums
        acc += tl.dot(qk, v, allow_tf32=True)

        # Update m_i and l_i
        m_i = m_i_new
        l_i = l_i_new

        # Advance pointers
        Q_block_ptr = tl.advance(Q_block_ptr, (0, BLOCK_K))
        K_block_ptr = tl.advance(K_block_ptr, (BLOCK_K, 0))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_K, 0))

    # Write back the result
    tl.store(O_block_ptr, acc)

import torch

def forward(q, k, v, q_scale, k_scale, BLOCK_M=128, BLOCK_N=128, BLOCK_K=32):
    M, K = q.shape
    N = k.shape[1]
    assert k.shape[0] == K and v.shape[0] == K and v.shape[1] == N

    o = torch.empty((M, N), device=q.device, dtype=q.dtype)

    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)

    _attn_fwd[grid](
        q, k, v, o,
        q_scale, k_scale,
        q.stride(0), q.stride(1),
        k.stride(0), k.stride(1),
        v.stride(0), v.stride(1),
        o.stride(0), o.stride(1),
        M, N, K, BLOCK_M, BLOCK_N, BLOCK_K
    )

    return o
