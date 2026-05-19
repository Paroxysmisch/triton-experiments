import triton
import triton.language as tl
import torch

@triton.jit
def rms_matmul_rbe(
    x_ptr: tl.pointer_type,
    w_ptr: tl.pointer_type,
    rms_w_ptr: tl.pointer_type,
    y_ptr: tl.pointer_type,
    x_batch_stride: tl.uint32,
    x_row_stride: tl.uint32,
    x_col_stride: tl.uint32,
    w_row_stride: tl.uint32,
    w_col_stride: tl.uint32,
    y_batch_stride: tl.uint32,
    y_row_stride: tl.uint32,
    y_col_stride: tl.uint32,
    B: tl.uint32,
    M: tl.uint32,
    N: tl.uint32,
    K: tl.uint32,
    THETA: tl.float32,
    APPLY_ROTARY: tl.int32,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    EPS: tl.float32
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    x_ptrs = x_ptr + (offs_am[:, None] * x_row_stride + offs_k[None, :] * x_col_stride)
    w_ptrs = w_ptr + (offs_k[:, None] * w_row_stride + offs_bn[None, :] * w_col_stride)
    y_ptrs = y_ptr + (offs_am[:, None] * y_row_stride + offs_bn[None, :] * y_col_stride)

    rms_w_ptrs = rms_w_ptr + tl.arange(0, BLOCK_SIZE_M)

    # Load RMS weights
    rms_w = tl.load(rms_w_ptrs, mask=offs_am < M, other=1.0)

    # Compute RMS normalization
    x = tl.load(x_ptrs, mask=offs_am[:, None] < M, other=0.0)
    x_squared = x * x
    x_squared_sum = tl.sum(x_squared, axis=1)
    x_rms = tl.sqrt(x_squared_sum / K + EPS)
    x_normalized = x / x_rms[:, None]

    # Apply RMS normalization
    x_normalized = x_normalized * rms_w[:, None]

    # Optionally apply rotary embeddings
    if APPLY_ROTARY:
        theta = THETA * (offs_am % M)
        cos_theta = tl.cos(theta)
        sin_theta = tl.sin(theta)
        x_normalized = x_normalized * cos_theta[:, None] + tl.rot90(x_normalized, k=1, dims=(0, 1)) * sin_theta[:, None]

    # Matrix multiplication
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        w = tl.load(w_ptrs, mask=offs_k[:, None] < K, other=0.0)
        x_normalized = tl.load(x_normalized, mask=offs_am[:, None] < M, other=0.0)
        acc += tl.dot(x_normalized, w)

    # Store the result
    tl.store(y_ptrs, acc, mask=offs_am[:, None] < M)

def rms_matmul_rbe_wrapper(x, w, rms_w, theta, apply_rotary):
    B, M, K = x.shape
    N = w.shape[1]

    assert x.is_cuda and w.is_cuda and rms_w.is_cuda, "Expected CUDA tensors"
    assert x.is_contiguous() and w.is_contiguous() and rms_w.is_contiguous(), "Expected contiguous tensors"

    y = torch.empty((B, M, N), device=x.device, dtype=x.dtype)

    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 16
    GROUP_SIZE_M = 8
    EPS = 1e-9

    rms_matmul_rbe[(B * M * N) // (BLOCK_SIZE_M * BLOCK_SIZE_N)](
        x, w, rms_w, y,
        x.stride(0), x.stride(1), x.stride(2),
        w.stride(0), w.stride(1),
        y.stride(0), y.stride(1), y.stride(2),
        B, M, N, K, theta, int(apply_rotary),
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M, EPS,
        num_warps=4
    )

    return y
