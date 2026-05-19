import triton
import triton.language as tl

# Define block sizes
BLOCK_M = 128  # Block size for query dimension
BLOCK_DMODEL = 64  # Block size for model dimension
BLOCK_N = 128  # Block size for key/value dimension

@triton.jit
def _fwd_kernel(Q, K, V, Out, stride_qm, stride_qk, stride_km, stride_kn, stride_vm, stride_vn, stride_om, stride_on, Lk, scale, BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr):
    # Compute program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Define ranges for blocks
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Load Q, K, V blocks
    Q_block = tl.load(Q + rm[:, None] * stride_qm + tl.arange(0, BLOCK_DMODEL) * stride_qk)
    K_block = tl.load(K + rn[None, :] * stride_kn + tl.arange(0, BLOCK_DMODEL) * stride_km)
    V_block = tl.load(V + rn[None, :] * stride_vn + tl.arange(0, BLOCK_DMODEL) * stride_vm)

    # Compute scaled dot product
    QK = tl.dot(Q_block, K_block, trans_b=True)
    QK_scaled = QK * scale

    # Apply softmax to the scaled dot product
    QK_softmax = tl.softmax(QK_scaled, axis=1)

    # Compute attention output
    Out_block = tl.dot(QK_softmax, V_block)

    # Store the result
    tl.store(Out + rm[:, None] * stride_om + rn[None, :] * stride_on, Out_block)

def context_attention_fwd(Q, K, V, Lk, BLOCK_M=BLOCK_M, BLOCK_DMODEL=BLOCK_DMODEL, BLOCK_N=BLOCK_N):
    # Check dimensions
    assert Q.shape[1] == K.shape[1] == V.shape[1] == Lk, "Mismatch in head dimension Lk"
    
    # Compute scale factor
    scale = 1.0 / (Q.shape[1] ** 0.5)

    # Prepare output tensor
    Out = torch.empty((Q.shape[0], V.shape[0]), device=Q.device, dtype=Q.dtype)

    # Define grid size
    grid = (triton.cdiv(Q.shape[0], BLOCK_M), triton.cdiv(V.shape[0], BLOCK_N))

    # Launch the kernel
    _fwd_kernel[grid](
        Q, K, V, Out,
        Q.stride(0), Q.stride(1),
        K.stride(0), K.stride(1),
        V.stride(0), V.stride(1),
        Out.stride(0), Out.stride(1),
        Lk, scale,
        BLOCK_M=BLOCK_M, BLOCK_DMODEL=BLOCK_DMODEL, BLOCK_N=BLOCK_N
    )

    return Out
