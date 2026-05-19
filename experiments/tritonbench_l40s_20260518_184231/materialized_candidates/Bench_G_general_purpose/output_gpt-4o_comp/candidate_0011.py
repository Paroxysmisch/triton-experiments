import triton
import triton.language as tl

# Define the Triton kernel
@triton.jit
def rms_matmul_rbe(
    x_ptr, w_ptr, rms_w_ptr, out_ptr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    THETA: tl.constexpr, APPLY_ROTARY: tl.constexpr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr
):
    # Calculate block indices
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Calculate the start indices of the block
    start_m = pid_m * BLOCK_SIZE_M
    start_n = pid_n * BLOCK_SIZE_N

    # Initialize accumulators for the output block
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Iterate over the K dimension
    for k in range(0, K, BLOCK_SIZE_K):
        # Load blocks of x and w
        x_block = tl.load(x_ptr + (start_m + tl.arange(0, BLOCK_SIZE_M))[:, None] * K + (k + tl.arange(0, BLOCK_SIZE_K)), mask=(start_m + tl.arange(0, BLOCK_SIZE_M))[:, None] < M)
        w_block = tl.load(w_ptr + (k + tl.arange(0, BLOCK_SIZE_K))[:, None] * N + (start_n + tl.arange(0, BLOCK_SIZE_N)), mask=(start_n + tl.arange(0, BLOCK_SIZE_N)) < N)

        # Apply RMS normalization to x_block
        rms = tl.sqrt(tl.sum(x_block * x_block, axis=1) / BLOCK_SIZE_K)
        rms = rms[:, None]  # Broadcast to match x_block dimensions
        x_block_normalized = x_block / rms

        # Perform the matrix multiplication
        acc += tl.dot(x_block_normalized, w_block)

    # Optionally apply rotary embeddings
    if APPLY_ROTARY:
        pos = tl.arange(0, BLOCK_SIZE_N)
        angle = pos * THETA
        cos_angle = tl.cos(angle)
        sin_angle = tl.sin(angle)
        acc = acc * cos_angle + acc * sin_angle

    # Store the result
    tl.store(out_ptr + (start_m + tl.arange(0, BLOCK_SIZE_M))[:, None] * N + (start_n + tl.arange(0, BLOCK_SIZE_N)), acc, mask=(start_m + tl.arange(0, BLOCK_SIZE_M))[:, None] < M)

# Define the wrapper function
def rms_matmul_rbe_wrapper(x, w, rms_w, THETA, APPLY_ROTARY, BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=32):
    import torch

    # Extract dimensions
    B, M, K = x.shape
    _, N, _ = w.shape

    # Allocate output tensor
    out = torch.empty((B, M, N), dtype=x.dtype, device=x.device)

    # Launch the Triton kernel
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N), B)
    rms_matmul_rbe[grid](
        x, w, rms_w, out,
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
        THETA=THETA, APPLY_ROTARY=APPLY_ROTARY,
        M=M, N=N, K=K
    )

    return out
