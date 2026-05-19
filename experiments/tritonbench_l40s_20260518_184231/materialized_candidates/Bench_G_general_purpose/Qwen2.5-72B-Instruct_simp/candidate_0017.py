import triton
import triton.language as tl

# Constants
BLOCK_M = 128
BLOCK_N = 64
HEAD_DIM = 64
STAGE = 2

# Inner computation function
@triton.jit
def _attn_fwd_inner(
    Q, K, V, Out,
    Q_scale, K_scale,
    stride_qm, stride_qh, stride_kn, stride_kh, stride_vm, stride_vh, stride_om, stride_oh,
    M, N, H, head_dim, stage,
    sm_scale: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr
):
    # Compute block indices
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_h = tl.program_id(2)

    # Compute offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_h = pid_h * HEAD_DIM + tl.arange(0, HEAD_DIM)

    # Initialize pointers
    Q_ptr = Q + (offs_m[:, None] * stride_qm + offs_h[None, :] * stride_qh)
    K_ptr = K + (offs_n[:, None] * stride_kn + offs_h[None, :] * stride_kh)
    V_ptr = V + (offs_n[:, None] * stride_vm + offs_h[None, :] * stride_vh)
    Out_ptr = Out + (offs_m[:, None] * stride_om + offs_h[None, :] * stride_oh)

    # Load Q, K, V
    q = tl.load(Q_ptr, mask=offs_m[:, None] < M, other=0.0)
    k = tl.load(K_ptr, mask=offs_n[:, None] < N, other=0.0)
    v = tl.load(V_ptr, mask=offs_n[:, None] < N, other=0.0)

    # Scale Q and K
    q = q * Q_scale
    k = k * K_scale

    # Compute attention scores
    logits = tl.dot(q, k, trans_b=True) * sm_scale

    # Apply softmax
    logits = tl.softmax(logits, axis=1)

    # Compute output
    out = tl.dot(logits, v)

    # Store output
    tl.store(Out_ptr, out, mask=offs_m[:, None] < M)

# Main kernel
@triton.jit
def _attn_fwd(
    Q, K, V, Out,
    Q_scale, K_scale,
    stride_qm, stride_qh, stride_kn, stride_kh, stride_vm, stride_vh, stride_om, stride_oh,
    M, N, H, head_dim,
    sm_scale: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr, STAGE: tl.constexpr
):
    # Compute grid size
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    grid_h = H

    # Launch inner kernel
    _attn_fwd_inner[grid_m, grid_n, grid_h](
        Q, K, V, Out,
        Q_scale, K_scale,
        stride_qm, stride_qh, stride_kn, stride_kh, stride_vm, stride_vh, stride_om, stride_oh,
        M, N, H, head_dim, STAGE,
        sm_scale, BLOCK_M, BLOCK_N, HEAD_DIM
    )

# Wrapper function
def forward(Q, K, V, Q_scale, K_scale, sm_scale, M, N, H, head_dim):
    # Initialize output tensor
    Out = tl.zeros((M, H, head_dim), dtype=Q.dtype)

    # Compute strides
    stride_qm = Q.stride(0)
    stride_qh = Q.stride(1)
    stride_kn = K.stride(0)
    stride_kh = K.stride(1)
    stride_vm = V.stride(0)
    stride_vh = V.stride(1)
    stride_om = Out.stride(0)
    stride_oh = Out.stride(1)

    # Compute grid size
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    grid_h = H

    # Launch kernel
    _attn_fwd[grid_m, grid_n, grid_h](
        Q, K, V, Out,
        Q_scale, K_scale,
        stride_qm, stride_qh, stride_kn, stride_kh, stride_vm, stride_vh, stride_om, stride_oh,
        M, N, H, head_dim,
        sm_scale, BLOCK_M, BLOCK_N, HEAD_DIM, STAGE
    )

    return Out
