import triton
import triton.language as tl

# Define constants for block sizes and model dimension
BLOCK_M = 128
BLOCK_N = 128
BLOCK_DMODEL = 64

@triton.jit
def _score_kernel(Q, K, M, Out, stride_qm, stride_qk, stride_kn, stride_om, stride_on, scale, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr):
    # Get the program ID
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Calculate the starting indices for the blocks
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    # Load blocks of Q and K
    Q_block = tl.load(Q + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk)
    K_block = tl.load(K + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_qk)

    # Compute the dot product
    score = tl.dot(Q_block, K_block)

    # Apply the scale and mask
    score = score * scale
    mask = tl.load(M + offs_m[:, None] * stride_qm + offs_n[None, :] * stride_on)
    score = tl.where(mask, score, float('-inf'))

    # Store the result
    tl.store(Out + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on, score)

def get_score(Q, K, M, scale):
    # Get dimensions
    num_queries, d_model = Q.shape
    num_keys, _ = K.shape

    # Define grid size
    grid = (triton.cdiv(num_queries, BLOCK_M), triton.cdiv(num_keys, BLOCK_N))

    # Allocate output tensor
    Out = torch.empty((num_queries, num_keys), device=Q.device, dtype=Q.dtype)

    # Calculate strides
    stride_qm = Q.stride(0)
    stride_qk = Q.stride(1)
    stride_kn = K.stride(1)
    stride_om = Out.stride(0)
    stride_on = Out.stride(1)

    # Launch the kernel
    _score_kernel[grid](
        Q, K, M, Out,
        stride_qm, stride_qk, stride_kn, stride_om, stride_on,
        scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=BLOCK_DMODEL
    )

    return Out
