import triton
import triton.language as tl

# Constants for block processing
BLOCK_M = 128
BLOCK_N = 128
HEAD_DIM = 64
STAGE = 2  # Number of stages for handling large sequences

@triton.jit
def _attn_fwd_inner(Q, K, V, Out, Q_scale, K_scale, stride_qz, stride_qh, stride_qm,
                    stride_kz, stride_kh, stride_kn, stride_vz, stride_vh, stride_vm,
                    stride_oz, stride_oh, stride_om, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr):
    # Program ID and offsets
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_h = tl.program_id(2)

    # Offsets for query, key, value, and output
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_h = pid_h

    # Load Q, K, V blocks
    Q_block = tl.load(Q + offs_h * stride_qh + offs_m[:, None] * stride_qm)
    K_block = tl.load(K + offs_h * stride_kh + offs_n[None, :] * stride_kn)
    V_block = tl.load(V + offs_h * stride_vh + offs_n[None, :] * stride_vm)

    # Scale and compute attention scores
    Q_scaled = Q_block * Q_scale
    K_scaled = K_block * K_scale
    attn_scores = tl.dot(Q_scaled, K_scaled.T)

    # Softmax normalization
    attn_scores = attn_scores - tl.max(attn_scores, axis=1, keepdims=True)
    attn_weights = tl.exp(attn_scores)
    attn_weights = attn_weights / tl.sum(attn_weights, axis=1, keepdims=True)

    # Compute output
    Out_block = tl.dot(attn_weights, V_block)

    # Store results
    tl.store(Out + offs_h * stride_oh + offs_m[:, None] * stride_om, Out_block)

@triton.jit
def _attn_fwd(Q, K, V, Out, Q_scale, K_scale, stride_qz, stride_qh, stride_qm,
              stride_kz, stride_kh, stride_kn, stride_vz, stride_vh, stride_vm,
              stride_oz, stride_oh, stride_om, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr):
    # Launch the inner kernel in a grid of blocks
    grid = (tl.cdiv(Q.shape[0], BLOCK_M), tl.cdiv(K.shape[1], BLOCK_N), Q.shape[1])
    _attn_fwd_inner[grid](Q, K, V, Out, Q_scale, K_scale, stride_qz, stride_qh, stride_qm,
                          stride_kz, stride_kh, stride_kn, stride_vz, stride_vh, stride_vm,
                          stride_oz, stride_oh, stride_om, BLOCK_M, BLOCK_N, HEAD_DIM)

def forward(Q, K, V, Q_scale, K_scale):
    # Determine output dimensions and allocate output tensor
    Z, H, M, N = Q.shape[0], Q.shape[1], Q.shape[2], K.shape[2]
    Out = torch.empty((Z, H, M, HEAD_DIM), device=Q.device, dtype=Q.dtype)

    # Strides for each dimension
    stride_qz, stride_qh, stride_qm = Q.stride(0), Q.stride(1), Q.stride(2)
    stride_kz, stride_kh, stride_kn = K.stride(0), K.stride(1), K.stride(2)
    stride_vz, stride_vh, stride_vm = V.stride(0), V.stride(1), V.stride(2)
    stride_oz, stride_oh, stride_om = Out.stride(0), Out.stride(1), Out.stride(2)

    # Launch the Triton kernel
    grid = (tl.cdiv(M, BLOCK_M), tl.cdiv(N, BLOCK_N), H)
    _attn_fwd[grid](Q, K, V, Out, Q_scale, K_scale, stride_qz, stride_qh, stride_qm,
                    stride_kz, stride_kh, stride_kn, stride_vz, stride_vh, stride_vm,
                    stride_oz, stride_oh, stride_om, BLOCK_M, BLOCK_N, HEAD_DIM)

    return Out
