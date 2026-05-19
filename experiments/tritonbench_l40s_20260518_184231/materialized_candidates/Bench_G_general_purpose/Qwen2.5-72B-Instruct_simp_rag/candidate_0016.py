import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel(
    Q, K, V, Out,
    B_Start_Loc, B_Seqlen,
    sm_scale,
    stride_qb, stride_qh, stride_qd,
    stride_kb, stride_kh, stride_kd,
    stride_vb, stride_vh, stride_vd,
    stride_ob, stride_oh, stride_od,
    B, H, N_CTX,
    BLOCK: tl.constexpr
):
    b_idx = tl.program_id(0)
    h_idx = tl.program_id(1)
    start_m = tl.program_id(2)

    offs_b = b_idx
    offs_h = h_idx
    offs_m = start_m * BLOCK + tl.arange(0, BLOCK)
    offs_n = tl.arange(0, BLOCK)
    Q_block_ptr = tl.make_block_ptr(
        base=Q, shape=(B, H, N_CTX, N_CTX), strides=(stride_qb, stride_qh, stride_qd),
        offsets=(offs_b, offs_h, 0, 0), block_shape=(1, 1, BLOCK, N_CTX), order=(3, 2, 1, 0)
    )
    K_block_ptr = tl.make_block_ptr(
        base=K, shape=(B, H, N_CTX, N_CTX), strides=(stride_kb, stride_kh, stride_kd),
        offsets=(offs_b, offs_h, 0, 0), block_shape=(1, 1, N_CTX, BLOCK), order=(3, 2, 1, 0)
    )
    V_block_ptr = tl.make_block_ptr(
        base=V, shape=(B, H, N_CTX, N_CTX), strides=(stride_vb, stride_vh, stride_vd),
        offsets=(offs_b, offs_h, 0, 0), block_shape=(1, 1, BLOCK, N_CTX), order=(3, 2, 1, 0)
    )
    Out_block_ptr = tl.make_block_ptr(
        base=Out, shape=(B, H, N_CTX, N_CTX), strides=(stride_ob, stride_oh, stride_od),
        offsets=(offs_b, offs_h, 0, 0), block_shape=(1, 1, BLOCK, N_CTX), order=(3, 2, 1, 0)
    )

    q = tl.load(Q_block_ptr, boundary_check=(2, 3))
    k = tl.load(K_block_ptr, boundary_check=(2, 3))
    v = tl.load(V_block_ptr, boundary_check=(2, 3))

    qk = tl.dot(q, k, allow_tf32=False) * sm_scale

    # Apply sliding window attention mask
    start_loc = tl.load(B_Start_Loc + offs_b)
    seqlen = tl.load(B_Seqlen + offs_b)
    m_start = start_loc + offs_m
    m_end = start_loc + seqlen
    mask = (m_start[:, None] <= m_end[None, :]) & (m_start[:, None] >= m_end[None, :] - BLOCK)
    qk = tl.where(mask, qk, float('-inf'))

    # Numerically stable softmax
    m = tl.max(qk, 1)
    qk = qk - m[:, None]
    p = tl.exp(qk)
    p_sum = tl.sum(p, 1)
    p = p / p_sum[:, None]

    # Compute output
    out = tl.dot(p, v, allow_tf32=False)

    # Write back to output
    tl.store(Out_block_ptr, out, boundary_check=(2, 3))

def context_attention_fwd(
    Q, K, V, sm_scale, B_Start_Loc, B_Seqlen, Out
):
    B, H, N_CTX, D_HEAD = Q.shape
    BLOCK = 128  # Block size for parallel execution

    # Configure grid and number of warps
    grid = (B, H, (N_CTX + BLOCK - 1) // BLOCK)
    num_warps = 4  # Adjust based on input sizes and GPU architecture

    # Launch the kernel
    _fwd_kernel[grid, num_warps](
        Q, K, V, Out,
        B_Start_Loc, B_Seqlen,
        sm_scale,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        Out.stride(0), Out.stride(1), Out.stride(2),
        B, H, N_CTX,
        BLOCK
    )
