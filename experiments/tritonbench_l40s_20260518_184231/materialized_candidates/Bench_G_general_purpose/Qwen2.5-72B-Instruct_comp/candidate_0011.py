import triton
import triton.language as tl

@triton.jit
def rms_matmul_rbe(
    x_ptr,  # pointer to the input tensor
    w_ptr,  # pointer to the transposed weight matrix
    rms_w_ptr,  # pointer to the auxiliary RMS weight
    y_ptr,  # pointer to the output tensor
    x_stride_m, x_stride_n,  # strides for the input tensor
    w_stride_m, w_stride_n,  # strides for the weight matrix
    y_stride_m, y_stride_n,  # strides for the output tensor
    M, N, K,  # dimensions of the matrices
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    APPLY_ROTARY: tl.constexpr,
    THETA: tl.constexpr
):
    # Program ID
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = num_pid_m * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * num_pid_m
    group_size_m = num_pid_m
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_n)

    # Block bounds
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Offsets for x and w
    x_ptrs = x_ptr + (offs_m[:, None] * x_stride_m + offs_k[None, :] * x_stride_n)
    w_ptrs = w_ptr + (offs_k[:, None] * w_stride_m + offs_n[None, :] * w_stride_n)

    # Load x and w
    x = tl.load(x_ptrs, mask=offs_m[:, None] < M)
    w = tl.load(w_ptrs, mask=offs_n[None, :] < N)

    # RMS normalization
    rms_w = tl.load(rms_w_ptr + offs_m)
    sum_of_squares = tl.sum(x * x, axis=1)
    mean = tl.sum(sum_of_squares, axis=0) / K
    rms = tl.sqrt(mean + 1e-6)
    x_normalized = x / rms[:, None]

    # Apply rotary embeddings if specified
    if APPLY_ROTARY:
        angle = THETA * (offs_m % K)
        cos_theta = tl.cos(angle)
        sin_theta = tl.sin(angle)
        x_normalized = x_normalized * cos_theta[:, None] + tl.roll(x_normalized, 1, axis=1) * sin_theta[:, None]

    # Matrix multiplication
    y = tl.dot(x_normalized, w)

    # Offsets for y
    y_ptrs = y_ptr + (offs_m[:, None] * y_stride_m + offs_n[None, :] * y_stride_n)

    # Store the result
    tl.store(y_ptrs, y, mask=offs_m[:, None] < M)

import torch
import triton
import triton.language as tl

def rms_matmul_rbe_wrapper(x, w, rms_w, apply_rotary=False, theta=0.0):
    # Ensure inputs are on the same device
    assert x.device == w.device == rms_w.device, "All inputs must be on the same device"
    device = x.device

    # Ensure inputs are of the correct type
    assert x.dtype == w.dtype == rms_w.dtype, "All inputs must be of the same data type"
    dtype = x.dtype

    # Get dimensions
    M, K = x.shape
    N, _ = w.shape

    # Allocate output tensor
    y = torch.empty((M, N), device=device, dtype=dtype)

    # Define grid and block sizes
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 16
    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),)

    # Launch the kernel
    rms_matmul_rbe[grid](
        x, w, rms_w, y,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
        y.stride(0), y.stride(1),
        M, N, K,
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
        apply_rotary, theta
    )

    return y
