import triton
import triton.language as tl

# Constants for block sizes
BLOCK_M = 128
BLOCK_N = 128
BLOCK_DMODEL = 64

# Main kernel function
@triton.jit
def _attn_fwd(
    Q, K, V, M, Out,
    stride_qm, stride_kn, stride_vn, stride_om,
    stride_qb, stride_kb, stride_vb, stride_ob,
    nheads, nctx, qk_scale, window_size: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr
):
    # Compute pointers for Q, K, V, and M
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_b = tl.program_id(3)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    Q_block_ptr = tl.make_block_ptr(
        base=Q + pid_b * stride_qb + pid_h * stride_qm,
        shape=(nctx, BLOCK_DMODEL),
        strides=(stride_qm, 1),
        offsets=(offs_m, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + pid_b * stride_kb + pid_h * stride_kn,
        shape=(BLOCK_DMODEL, nctx),
        strides=(1, stride_kn),
        offsets=(0, offs_n),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1)
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + pid_b * stride_vb + pid_h * stride_vn,
        shape=(nctx, BLOCK_DMODEL),
        strides=(stride_vn, 1),
        offsets=(offs_n, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0)
    )
    M_block_ptr = tl.make_block_ptr(
        base=M + pid_b * stride_ob + pid_h * stride_om,
        shape=(nctx, nctx),
        strides=(stride_om, 1),
        offsets=(offs_m, offs_n),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0)
    )
    Out_block_ptr = tl.make_block_ptr(
        base=Out + pid_b * stride_ob + pid_h * stride_om,
        shape=(nctx, BLOCK_DMODEL),
        strides=(stride_om, 1),
        offsets=(offs_m, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )

    # Initialize accumulators
    acc = tl.zeros((BLOCK_M, BLOCK_DMODEL), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    m_i = tl.full((BLOCK_M,), float('-inf'), dtype=tl.float32)

    # Compute QK
    Q_block = tl.load(Q_block_ptr)
    K_block = tl.load(K_block_ptr)
    QK = tl.dot(Q_block, K_block, allow_tf32=True)

    # Apply mask if sliding window attention is specified
    if window_size > 0:
        mask = tl.where(tl.abs(offs_m[:, None] - offs_n[None, :]) < window_size, 0, float('-inf'))
        QK += mask

    # Compute max and likelihoods
    qk_scale = tl.full((1,), qk_scale, dtype=tl.float32)
    QK *= qk_scale
    m_i = tl.max(QK, 1)
    QK = QK - m_i[:, None]
    P = tl.exp2(QK)

    # Update running maxima and likelihoods
    l_i_new = tl.sum(P, 1)
    l_i = l_i + l_i_new
    P = P / l_i[:, None]

    # Compute output
    V_block = tl.load(V_block_ptr)
    acc += tl.dot(P, V_block, allow_tf32=True)

    # Store results
    tl.store(Out_block_ptr, acc.to(Out.dtype.element_ty))
    tl.store(M_block_ptr, m_i.to(M.dtype.element_ty))

# Wrapper function
def _forward(Q, K, V, M, Out, nheads, nctx, qk_scale, window_size):
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_DMODEL = 64

    # Get tensor shapes
    B, H, M, D = Q.shape
    N = K.shape[2]

    # Configure grid
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N), H, B)

    # Launch kernel
    _attn_fwd[grid](
        Q, K, V, M, Out,
        Q.stride(2), K.stride(2), V.stride(2), Out.stride(2),
        Q.stride(0), K.stride(0), V.stride(0), Out.stride(0),
        nheads, nctx, qk_scale, window_size,
        BLOCK_M, BLOCK_N, BLOCK_DMODEL
    )

# Example usage
import torch

# Dummy data
B, H, M, D = 2, 8, 1024, 64
Q = torch.randn((B, H, M, D), device='cuda')
K = torch.randn((B, H, M, D), device='cuda')
V = torch.randn((B, H, M, D), device='cuda')
M = torch.zeros((B, H, M, M), device='cuda')
Out = torch.zeros((B, H, M, D), device='cuda')

# Parameters
nheads = H
nctx = M
qk_scale = 1.0 / (D ** 0.5)
window_size = 128

# Forward pass
_forward(Q, K, V, M, Out, nheads, nctx, qk_scale, window_size)

print(Out)
