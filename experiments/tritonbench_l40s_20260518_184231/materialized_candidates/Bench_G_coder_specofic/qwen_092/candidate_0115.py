triton
import triton
import triton.language as tl

@triton.jit
def rmsnorm_triton(
    x_ptr: tl.tensor,  # Input tensor [batch, M, K]
    rms_w_ptr: tl.tensor,  # RMS weights [batch, M]
    out_ptr: tl.tensor,  # Output tensor [batch, M, K]
    N_SIZE: tl.constexpr,  # Size of K dimension
    eps: tl.constexpr,  # Small constant to prevent division by zero
    BLOCK_M_SIZE: tl.constexpr,  # Block size in the M dimension
    BLOCK_N_SIZE: tl.constexpr,  # Block size in the N dimension
    num_warps: tl.constexpr  # Number of warps per block
):
    pid = tl.program_id(axis=0)
    bid = pid // (BLOCK_M_SIZE * BLOCK_N_SIZE)
    tid = pid % (BLOCK_M_SIZE * BLOCK_N_SIZE)
    m = bid * BLOCK_M_SIZE + tid // BLOCK_N_SIZE
    n = bid * BLOCK_N_SIZE + tid % BLOCK_N_SIZE

    x = tl.load(x_ptr + m * N_SIZE * BLOCK_N_SIZE + n, mask=m < x_ptr.shape[0] and n < x_ptr.shape[2], boundary_check=False)
    rms_w = tl.load(rms_w_ptr + m, mask=m < rms_w_ptr.shape[0], boundary_check=False)

    if m >= x_ptr.shape[0] or n >= x_ptr.shape[2]:
        tl.store(out_ptr + m * N_SIZE * BLOCK_N_SIZE + n, 0.0, mask=False)
        return

    sum_sq = 0.0
    for k in range(N_SIZE):
        sum_sq += x[k] * x[k]
    rms = tl.sqrt(sum_sq / N_SIZE + eps)
    normalized_x = x / rms
    scaled_x = normalized_x * rms_w

    tl.store(out_ptr + m * N_SIZE * BLOCK_N_SIZE + n, scaled_x)
