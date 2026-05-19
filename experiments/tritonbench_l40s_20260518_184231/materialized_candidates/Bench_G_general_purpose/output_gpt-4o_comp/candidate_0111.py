import triton
import triton.language as tl
import torch

# Define the Triton kernel for the inner attention forward pass
@triton.jit
def _attn_fwd_inner(Q, K, V, Out, stride_qz, stride_qh, stride_qm, stride_kz, stride_kh, stride_kn,
                    stride_vz, stride_vh, stride_vn, stride_oz, stride_oh, stride_om, qk_scale, 
                    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr):
    # Get program ID
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Create block pointers for Q, K, V, and Out
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    Q_ptrs = Q + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qh
    K_ptrs = K + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kh
    V_ptrs = V + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vh
    Out_ptrs = Out + offs_m[:, None] * stride_om + offs_d[None, :] * stride_oh

    # Load Q and K, perform QK^T
    Q_block = tl.load(Q_ptrs)
    K_block = tl.load(K_ptrs)
    QK = tl.dot(Q_block, K_block.T) * qk_scale

    # Apply softmax
    m_i = tl.max(QK, axis=1)
    QK = tl.exp2(QK - m_i[:, None])
    l_i = tl.sum(QK, axis=1)

    # Normalize and compute output
    V_block = tl.load(V_ptrs)
    Out_block = tl.dot(QK, V_block)
    Out_block /= l_i[:, None]

    # Store the result
    tl.store(Out_ptrs, Out_block)

# Define the outer function to manage memory and block operations
def _attn_fwd(Q, K, V, BLOCK_M, BLOCK_N, BLOCK_DMODEL, qk_scale):
    # Get shapes and strides
    B, H, M, D = Q.shape
    _, _, N, _ = K.shape
    _, _, _, Dv = V.shape

    # Allocate output tensor
    Out = torch.empty((B, H, M, Dv), device=Q.device, dtype=Q.dtype)

    # Launch the Triton kernel
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _attn_fwd_inner[grid](
        Q, K, V, Out,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        Out.stride(0), Out.stride(1), Out.stride(2),
        qk_scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=BLOCK_DMODEL
    )

    return Out

# Define the wrapper function for input preparation and kernel execution
def _forward(Q, K, V, qk_scale, block_m=128, block_n=128, block_dmodel=64):
    try:
        # Execute the forward pass
        return _attn_fwd(Q, K, V, block_m, block_n, block_dmodel, qk_scale)
    except triton.OutOfResources as e:
        # Handle out-of-resources error by reducing block sizes
        block_m //= 2
        block_n //= 2
        block_dmodel //= 2
        if block_m < 16 or block_n < 16 or block_dmodel < 16:
            raise RuntimeError("Cannot allocate enough resources on GPU.")
        return _forward(Q, K, V, qk_scale, block_m, block_n, block_dmodel)
