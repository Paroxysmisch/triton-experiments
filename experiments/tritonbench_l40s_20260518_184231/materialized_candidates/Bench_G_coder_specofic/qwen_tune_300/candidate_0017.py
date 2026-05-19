import torch
import triton
import triton.language as tl
from packaging import version

@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q, q_scale, K_ptrs, K_scale_ptr, V_ptrs, start_m, BLOCK_M, HEAD_DIM, BLOCK_N, STAGE, offs_m, offs_n, N_CTX):
    # Load K and apply scaling
    k = tl.load(K_ptrs + offs_n[:, None] * HEAD_DIM + offs_m[None, :] * 16, mask=offs_n[:, None] < BLOCK_N, other=0.0)
    k_scale = tl.load(K_scale_ptr + offs_n, mask=offs_n < BLOCK_N, other=0.0)
    qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    if STAGE == 1:
        qk += tl.dot(q, k)
    qk *= q_scale
    qk *= k_scale[None, :]

    # Causal mask
    qk += tl.where(offs_m[:, None] >= (start_m + offs_n[None, :]), 0, -1.0e6)

    # Compute m_ij, l_ij
    m_ij = tl.max(qk, 1)
    p = tl.exp(qk - m_ij[:, None])
    l_ij = tl.sum(p, 1)
    # Update m_i and l_i
    m_i_new = tl.maximum(m_i, m_ij)
    alpha = tl.exp(m_i - m_i_new)
    beta = tl.exp(m_ij - m_i_new)
    l_i_new = alpha * l_i + beta * l_ij
    # Update acc
    p_scale = beta / l_i_new
    p = p * p_scale[:, None]
    acc_scale = l_i / l_i_new * alpha
    acc = acc * acc_scale[:, None]
    # Update acc
    v = tl.load(V_ptrs + offs_n[:, None] * 16 + offs_m[None, :] * HEAD_DIM, mask=offs_n[:, None] < BLOCK_N, other=0.0)
    acc += tl.dot(p.to(HEAD_DIM), v)
    # Update m_i and l_i
    l_i = l_i_new
    m_i = m_i_new
    return acc, l_i, m_i

@triton.jit
def _attn_fwd(Q, K, V, Q_scale, K_scale, Out, stride_qz, stride_qh, stride_qm, stride_qk, stride_kz, stride_kh, stride_kn, stride_kk, stride_vz, stride_vh, stride_vk, stride_vn, stride_oz, stride_oh, stride_om, stride_on, Z, H, N_CTX, HEAD_DIM, BLOCK_M, BLOCK_N, STAGE):
    # Batch, head, block indices
    pid_z = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)
    pid_m = tl.program_id(axis=2)
    # Create index ranges
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # Start of index ranges
    start_m = pid_m * BLOCK_M
    # Create pointers for batch and head dimensions
    batch_offset_q = pid_z * stride_qz + pid_h * stride_qh
    batch_offset_k = pid_z * stride_kz + pid_h * stride_kh
    batch_offset_v = pid_z * stride_vz + pid_h * stride_vh
    batch_offset_o = pid_z * stride_oz + pid_h * stride_oh
    q_ptr = Q + batch_offset_q + offs_m[:, None] * stride_qm + offs_n[None, :] * stride_qk
    k_ptr = K + batch_offset_k + offs_n[:, None] * stride_kn + offs_m[None, :] * 16
    v_ptr = V + batch_offset_v + offs_n[:, None] * stride_vk + offs_m[None, :] * 16
    o_ptr = Out + batch_offset_o + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    # Load Q and compute scaling
    q = tl.load(q_ptr, mask=offs_m[:, None] < BLOCK_M, other=0.0)
    q_scale = tl.load(Q_scale + batch_offset_q + offs_m * stride_qm)
    # Initialize variables for attention
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    m_i_new = tl.zeros([BLOCK_M], dtype=tl.float32)
    # Invoke inner kernel for attention computation
    if STAGE == 2:
        tl.debug_barrier()
    acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, q_scale, k_ptr, K_scale + batch_offset_k + offs_n, v_ptr, start_m, BLOCK_M, HEAD_DIM, BLOCK_N, STAGE, offs_m, offs_n, N_CTX)
    # Store results
    acc = acc.to(HEAD_DIM)
    tl.store(o_ptr, acc, mask=offs_m[:, None] < BLOCK_M)
    # Update running statistics
    m_i = m_i.to(HEAD_DIM)
    l_i = l_i.to(HEAD_DIM)
    return

class _attention_v1(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, q_scale, k_scale):
        # Prepare outputs and constants
        HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        assert HEAD_DIM_Q == q.shape[-1] and HEAD_DIM_K == k.shape[-1]
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}
        o = torch.empty_like(q)
        BLOCK_M = 128 if HEAD_DIM_K <= 64 else 64
        # Compute settings
        num_stages = 3 if HEAD_DIM_K <= 64 else 4
        num_warps = 8
        grid = (q.shape[0], q.shape[1], triton.cdiv(q.shape[2], BLOCK_M))
        # Define scaling constants
        if version.parse(torch.__version__) >= version.parse("2.1.0"):
            q_scale_ = torch.full((triton.cdiv(q.shape[2], 128), q.shape[0], q.shape[1]), 1.0 / 128.0, device=q.device, dtype=torch.float32)
        else:
            q_scale_ = torch.full((triton.cdiv(q.shape[2], 128), q.shape[0], q.shape[1]), 1.0 / 128.0, dtype=torch.float32, device=q.device)
        # Launch Triton kernel
        _attn_fwd[grid](q, k, v, q_scale_, k_scale, o, q.stride(0), q.stride(1), q.stride(2), q.stride(3), k.stride(0), k.stride(1), k.stride(2), k.stride(3), v.stride(0), v.stride(1), v.stride(2), v.stride(3), o.stride(0), o.stride(1), o.stride(2), o.stride(3), q.shape[0], q.shape[1], q.shape[2], HEAD_DIM_K, BLOCK_M, HEAD_DIM_K, num_stages, num_warps, STAGE=2)
        ctx.save_for_backward(q, k, v, q_scale, k_scale)
        return o

def attention_v1(q, k, v, q_scale, k_scale):
    return _attention_v1.apply(q, k, v, q_scale, k_scale)
