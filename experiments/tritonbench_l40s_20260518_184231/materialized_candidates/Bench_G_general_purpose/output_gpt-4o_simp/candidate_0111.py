import triton
import triton.language as tl
import torch

# Constants for block sizes
_BLOCK_M = 128  # Block size for batch dimension
_BLOCK_N = 128  # Block size for sequence length dimension
_BLOCK_DMODEL = 64  # Block size for model dimension

@triton.jit
def _attn_fwd(
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    stride_qm, stride_qn, stride_qd,
    stride_km, stride_kn, stride_kd,
    stride_vm, stride_vn, stride_vd,
    stride_om, stride_on, stride_od,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    context_len: tl.constexpr, sliding_window: tl.constexpr
):
    # Block indices
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets for the blocks
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    # Pointers to the Q, K, V matrices
    Q = tl.load(Q_ptr + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd)
    K = tl.load(K_ptr + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd)
    V = tl.load(V_ptr + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd)

    # Compute attention scores (QK^T)
    QK = tl.dot(Q, K.T)

    # Apply sliding window mask if enabled
    if sliding_window:
        mask = (offs_n[:, None] - offs_m[None, :]) < context_len
        QK = tl.where(mask, QK, float('-inf'))

    # Softmax normalization
    QK_max = tl.max(QK, axis=1)
    QK_exp = tl.exp(QK - QK_max[:, None])
    QK_sum = tl.sum(QK_exp, axis=1)
    QK_softmax = QK_exp / QK_sum[:, None]

    # Compute attention output
    Out = tl.dot(QK_softmax, V)

    # Store the result
    tl.store(Out_ptr + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od, Out)

def _forward(Q, K, V, context_len, sliding_window=False):
    # Ensure inputs are contiguous
    Q = Q.contiguous()
    K = K.contiguous()
    V = V.contiguous()

    # Get shapes
    batch_size, seq_len, d_model = Q.shape

    # Allocate output tensor
    Out = torch.empty((batch_size, seq_len, d_model), device=Q.device, dtype=Q.dtype)

    # Define grid dimensions
    grid = (triton.cdiv(batch_size, _BLOCK_M), triton.cdiv(seq_len, _BLOCK_N))

    # Launch the Triton kernel
    _attn_fwd[grid](
        Q, K, V, Out,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        Out.stride(0), Out.stride(1), Out.stride(2),
        _BLOCK_M=_BLOCK_M, _BLOCK_N=_BLOCK_N, BLOCK_DMODEL=_BLOCK_DMODEL,
        context_len=context_len, sliding_window=sliding_window
    )

    return Out
