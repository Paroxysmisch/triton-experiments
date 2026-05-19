import triton
import triton.language as tl

@triton.jit
def softplus_linear_kernel(
    input_ptr,  # Pointer to the input tensor
    weight_ptr,  # Pointer to the weight tensor
    bias_ptr,  # Pointer to the bias tensor (optional)
    output_ptr,  # Pointer to the output tensor
    M,  # Number of rows in the input tensor
    N,  # Number of columns in the input tensor
    K,  # Number of columns in the weight tensor
    beta,  # Beta parameter for the Softplus function
    threshold,  # Threshold for numerical stability
    stride_im,  # Stride for input matrix
    stride_in,  # Stride for input matrix
    stride_wk,  # Stride for weight matrix
    stride_wk1,  # Stride for weight matrix
    stride_om,  # Stride for output matrix
    stride_on,  # Stride for output matrix
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_n)

    # Compute the block of the output matrix this program should compute
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = input_ptr + (offs_am[:, None] * stride_im + offs_k[None, :] * stride_in)
    b_ptrs = weight_ptr + (offs_k[:, None] * stride_wk + offs_bn[None, :] * stride_wk1)

    # Load the input and weight blocks
    a_mask = (offs_am[:, None] < M) & (offs_k[None, :] < K)
    b_mask = (offs_k[:, None] < K) & (offs_bn[None, :] < N)
    a = tl.load(a_ptrs, mask=a_mask, other=0.0)
    b = tl.load(b_ptrs, mask=b_mask, other=0.0)

    # Compute the dot product
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_in
        b_ptrs += BLOCK_SIZE_K * stride_wk

    # Add bias if provided
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + offs_bn, mask=offs_bn < N, other=0.0)
        acc += bias[None, :]

    # Apply the Softplus function
    x = acc * beta
    exp_x = tl.exp(x)
    softplus = (1 / beta) * tl.log(1 + exp_x)
    softplus = tl.where(x > threshold, acc, softplus)

    # Store the result
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    out_ptrs = output_ptr + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, softplus, mask=out_mask)

import torch
import triton
import triton.language as tl

def softplus_linear(input, weight, bias=None, beta=1, threshold=20):
    M, K = input.shape
    N, _ = weight.shape
    output = torch.empty((M, N), device=input.device, dtype=input.dtype)

    # Define grid and block sizes
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 16
    GROUP_SIZE_M = 8

    grid = (
        triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),
    )

    # Launch the kernel
    softplus_linear_kernel[grid](
        input, weight, bias, output,
        M, N, K,
        beta, threshold,
        input.stride(0), input.stride(1),
        weight.stride(0), weight.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M
    )

    return output
