import triton
import triton.language as tl

# Constants
BLOCK_M = 64
BLOCK_N = 64
BLOCK_DMODEL = 64

@triton.jit
def _attn_fwd_inner(Q, K, V, QK, m_i, l_i, scale, window_mask, max_seq_len):
    """
    Inner kernel to compute attention scores and apply softmax.
    """
    pid = tl.program_id(axis=0)
    row = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    col = tl.arange(0, BLOCK_N)

    # Load Q and K
    q = tl.load(Q + row[:, None] * BLOCK_DMODEL + col[None, :])
    k = tl.load(K + row[:, None] * BLOCK_DMODEL + col[None, :])

    # Compute QK
    qk = tl.dot(q, k.T) * scale

    # Apply window mask if provided
    if window_mask is not None:
        tl.store(QK + row[:, None] * BLOCK_N + col[None, :], qk * window_mask, mask=window_mask[row[:, None], col[None, :]])

    # Exponential scaling and softmax
    max_qk = tl.max(qk, axis=1, keepdim=True)
    qk_exp = tl.math.exp2(qk - max_qk)
    qk_sum = tl.sum(qk_exp, axis=1, keepdim=True)
    softmax = qk_exp / qk_sum

    # Update running maxima and likelihoods
    m_i[pid] = max_qk
    l_i[pid] = tl.log(qk_sum)

    # Store softmax
    tl.store(QK + row[:, None] * BLOCK_N + col[None, :], softmax)

@triton.jit
def _attn_fwd(Q, K, V, output, max_seq_len, window_mask=None):
    """
    Outer kernel to manage memory layout and block pointers.
    """
    pid = tl.program_id(axis=0)
    grid_m = tl.cdiv(Q.shape[0], BLOCK_M)
    grid_n = tl.cdiv(Q.shape[1], BLOCK_N)

    # Calculate block pointers
    block_m = pid // grid_n
    block_n = pid % grid_n

    # Initialize memory pointers
    Q_ptr = Q + block_m * BLOCK_M * BLOCK_DMODEL
    K_ptr = K + block_m * BLOCK_M * BLOCK_DMODEL
    V_ptr = V + block_m * BLOCK_M * BLOCK_DMODEL
    output_ptr = output + block_m * BLOCK_M * BLOCK_N

    # Initialize running maxima and likelihoods
    m_i = tl.zeros((), dtype=tl.float32)
    l_i = tl.zeros((), dtype=tl.float32)

    # Scale for softmax
    scale = 1.0 / (BLOCK_DMODEL ** 0.5)

    # Execute inner kernel
    _attn_fwd_inner(Q_ptr, K_ptr, V_ptr, output_ptr, m_i, l_i, scale, window_mask, max_seq_len)

@triton.jit
def _forward(Q, K, V, output, max_seq_len, window_mask=None):
    """
    Wrapper function to handle input preparation and kernel execution.
    """
    # Check if there are enough resources
    if triton.out_of_resources():
        raise triton.OutOfResources("Not enough resources to execute the kernel")

    # Execute the outer kernel
    grid = (triton.cdiv(Q.shape[0], BLOCK_M) * triton.cdiv(Q.shape[1], BLOCK_N),)
    _attn_fwd[grid](Q, K, V, output, max_seq_len, window_mask)

def forward(Q, K, V, output, max_seq_len, window_mask=None):
    """
    Public function to execute the attention mechanism.
    """
    # Ensure input shapes are compatible
    if Q.shape[0] != K.shape[0] or Q.shape[0] != V.shape[0] or Q.shape[1] != output.shape[1]:
        raise ValueError("Input shapes must match")

    # Initialize output tensor if not provided
    if output is None:
        output = tl.zeros((Q.shape[0], K.shape[1], V.shape[2]), dtype=Q.dtype)

    # Execute the forward pass
    _forward(Q, K, V, output, max_seq_len, window_mask)

    return output
