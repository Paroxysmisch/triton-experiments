import triton
import triton.language as tl

@triton.jit
def ff_llama(x_ptr, w1_ptr, w3_ptr, rms_w_ptr, output_ptr, M, N, K, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, EPS, USE_FP8: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = num_pid_m * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * num_pid_m
    first_pid_n = group_id * num_pid_n
    pid_m = first_pid_m + (pid % num_pid_m)
    pid_n = first_pid_n + (pid % num_pid_n)

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    x_ptrs = x_ptr + (offs_am[:, None] * K + offs_k[None, :]) * x_ptr.dtype.itemsize
    w1_ptrs = w1_ptr + (offs_k[:, None] * N + offs_bn[None, :]) * w1_ptr.dtype.itemsize
    w3_ptrs = w3_ptr + (offs_k[:, None] * N + offs_bn[None, :]) * w3_ptr.dtype.itemsize
    rms_w_ptrs = rms_w_ptr + offs_am * rms_w_ptr.dtype.itemsize

    acc1 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        x = tl.load(x_ptrs, mask=offs_k[None, :] < K - k, other=0.0)
        w1 = tl.load(w1_ptrs, mask=offs_k[:, None] < K - k, other=0.0)
        w3 = tl.load(w3_ptrs, mask=offs_k[:, None] < K - k, other=0.0)

        if USE_FP8:
            x = x.to(tl.float32)
            w1 = w1.to(tl.float32)
            w3 = w3.to(tl.float32)

        acc1 += tl.dot(x, w1)
        acc2 += tl.dot(x, w3)

        x_ptrs += BLOCK_SIZE_K * x_ptr.dtype.itemsize
        w1_ptrs += BLOCK_SIZE_K * w1_ptr.dtype.itemsize
        w3_ptrs += BLOCK_SIZE_K * w3_ptr.dtype.itemsize

    # RMS normalization
    rms_w = tl.load(rms_w_ptrs, mask=offs_am < M, other=1.0)
    l2_norm = tl.sqrt(tl.sum(acc1 * acc1, axis=1) / N + EPS)
    acc1 /= l2_norm[:, None]

    # Apply scaled sigmoid activation
    output = tl.sigmoid(acc1) * acc2 * rms_w[:, None]

    # Store the result
    output_ptrs = output_ptr + (offs_am[:, None] * N + offs_bn[None, :]) * output_ptr.dtype.itemsize
    tl.store(output_ptrs, output, mask=offs_bn[None, :] < N)

import torch

def kernel_ff(x, w1, w3, rms_w, EPS=1e-5, BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=32, USE_FP8=False):
    assert x.is_cuda and w1.is_cuda and w3.is_cuda and rms_w.is_cuda, "All tensors must be on CUDA"
    assert x.is_contiguous() and w1.is_contiguous() and w3.is_contiguous() and rms_w.is_contiguous(), "All tensors must be contiguous"
    assert x.shape[-1] == w1.shape[0] == w3.shape[0], "Dimension mismatch"
    assert rms_w.shape[0] == x.shape[0], "Dimension mismatch for RMS weights"

    M, K = x.shape
    N = w1.shape[1]

    # Transpose weight matrices
    w1 = w1.t().contiguous()
    w3 = w3.t().contiguous()

    # Allocate output tensor
    output = torch.empty((M, N), device=x.device, dtype=x.dtype)

    # Calculate grid size
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )

    # Launch the kernel
    ff_llama[grid](
        x, w1, w3, rms_w, output, M, N, K, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, EPS, USE_FP8
    )

    return output
