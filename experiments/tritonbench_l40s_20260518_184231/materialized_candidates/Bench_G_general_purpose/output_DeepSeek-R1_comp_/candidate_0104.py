import torch
import triton
import triton.language as tl

@triton.jit
def _attn_fwd_inner(
    acc, l_i, m_i, q,
    K_block_ptr, V_block_ptr,
    q_scale, k_scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
    N: tl.constexpr, D: tl.constexpr,
):
    # Load K and V blocks with proper masking
    k = tl.load(K_block_ptr, boundary_check=(0, 1))
    v = tl.load(V_block_ptr, boundary_check=(0, 1))
    k = tl.maximum(k, 0)  # Ensure no NaNs
    v = tl.maximum(v, 0)  # Ensure no NaNs

    # Compute scores
    scores = tl.dot(q, tl.trans(k.to(tl.float32))) * q_scale * k_scale
    scores = scores.to(tl.float32)

    # Compute current block max
    m_curr = tl.max(scores, axis=1)
    m_new = tl.maximum(m_i, m_curr)

    # Compute scaling factors
    alpha = tl.exp(m_i - m_new)
    beta = tl.exp(m_curr - m_new)

    # Compute exponentials and update sums
    current_exp = tl.exp(scores - m_new[:, None])
    sum_exp = tl.sum(current_exp, axis=1)
    l_i_new = alpha * l_i + sum_exp

    # Update accumulated values
    v_f32 = v.to(tl.float32)
    acc_contrib = tl.dot(current_exp, v_f32)
    acc_new = alpha[:, None] * acc + acc_contrib

    return acc_new, l_i_new, m_new

@triton.jit
def _attn_fwd(
    q_ptr, k_ptr, v_ptr, o_ptr,
    q_scale, k_scale,
    s_qb, s_qh, s_qm, s_qk,
    s_kb, s_kh, s_kn, s_kk,
    s_vb, s_vh, s_vn, s_vk,
    B, H, M, N, D,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_m = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    # Initialize Q pointer and load Q
    q_block_ptr = tl.make_block_ptr(
        base=q_ptr + pid_b * s_qb + pid_h * s_qh,
        shape=(M, D),
        strides=(s_qm, s_qk),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_D),
        order=(1, 0)
    )
    q = tl.load(q_block_ptr, boundary_check=(0, 1))

    # Initialize accumulators
    m_i = tl.full((BLOCK_M,), -float('inf'), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)

    # Loop over K and V blocks
    num_n_blocks = tl.cdiv(N, BLOCK_N)
    for n in range(num_n_blocks):
        start_n = n * BLOCK_N
        K_block_ptr = tl.make_block_ptr(
            base=k_ptr + pid_b * s_kb + pid_h * s_kh,
            shape=(N, D),
            strides=(s_kn, s_kk),
            offsets=(start_n, 0),
            block_shape=(BLOCK_N, BLOCK_D),
            order=(1, 0)
        )
        V_block_ptr = tl.make_block_ptr(
            base=v_ptr + pid_b * s_vb + pid_h * s_vh,
            shape=(N, D),
            strides=(s_vn, s_vk),
            offsets=(start_n, 0),
            block_shape=(BLOCK_N, BLOCK_D),
            order=(1, 0)
        )
        acc, l_i, m_i = _attn_fwd_inner(
            acc, l_i, m_i, q,
            K_block_ptr, V_block_ptr,
            q_scale, k_scale,
            BLOCK_M, BLOCK_N, BLOCK_D,
            N, D
        )

    # Write output
    o_block_ptr = tl.make_block_ptr(
        base=o_ptr + pid_b * s_qb + pid_h * s_qh,
        shape=(M, D),
        strides=(s_qm, s_qk),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_D),
        order=(1, 0)
    )
    tl.store(o_block_ptr, (acc / l_i[:, None]).to(tl.float16), boundary_check=(0, 1))

def forward(q, k, v, q_scale, k_scale):
    B, H, M, D = q.shape
    N = k.shape[2]
    o = torch.empty_like(q)

    grid = (B, H, triton.cdiv(M, 128))
    _attn_fwd[grid](
        q, k, v, o,
        q_scale, k_scale,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        B, H, M, N, D,
        BLOCK_M=128, BLOCK_N=64, BLOCK_D=64
    )
    return o
