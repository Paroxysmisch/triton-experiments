import triton
import triton.language as tl

# Constants defining block sizes
BLOCK_M = 128
BLOCK_N = 128

@triton.jit
def _attn_fwd_inner(Q, K, V, Out, Q_scale, K_scale, N_CTX, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # Define program ids
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Define offsets for the blocks
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Load Q and K blocks
    Q_block = tl.load(Q + offs_m[:, None] * N_CTX + offs_n[None, :])
    K_block = tl.load(K + offs_n[:, None] * N_CTX + offs_m[None, :])

    # Scale Q and K
    Q_block_scaled = Q_block * Q_scale
    K_block_scaled = K_block * K_scale

    # Compute scaled dot-product attention
    scores = tl.dot(Q_block_scaled, K_block_scaled)

    # Apply exponential function
    exp_scores = tl.exp(scores)

    # Initialize accumulators
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)

    # Accumulate scores and normalization factor
    acc += exp_scores
    l_i += tl.sum(exp_scores, axis=1)

    # Normalize and store the result in the output tensor
    Out_block = acc / l_i[:, None]
    tl.store(Out + offs_m[:, None] * N_CTX + offs_n[None, :], Out_block)

@triton.jit
def _attn_fwd(Q, K, V, Out, Q_scale, K_scale, N_CTX, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # Launch the kernel over blocks
    grid = (triton.cdiv(Q.shape[0], BLOCK_M), triton.cdiv(N_CTX, BLOCK_N))
    _attn_fwd_inner[grid](Q, K, V, Out, Q_scale, K_scale, N_CTX, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)

# Wrapper function to call the kernel
def attention_forward(Q, K, V, Q_scale, K_scale):
    # Assuming Q, K, V are Triton tensors
    N_CTX = Q.shape[1]
    Out = tl.zeros_like(Q)  # Output tensor

    # Call the kernel
    _attn_fwd(Q, K, V, Out, Q_scale, K_scale, N_CTX, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)

    return Out
