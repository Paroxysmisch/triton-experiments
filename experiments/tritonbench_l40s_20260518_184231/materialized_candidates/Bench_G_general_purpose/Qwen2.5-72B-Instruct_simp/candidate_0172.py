import triton
import triton.language as tl

@triton.jit
def log_softmax_kernel(
    X_ptr,  # Pointer to input matrix
    Y_ptr,  # Pointer to output matrix
    M,      # Number of rows
    N,      # Number of columns
    K,      # Auxiliary dimension (not used in this kernel)
    stride_xm,  # Stride for rows in input matrix
    stride_xn,  # Stride for columns in input matrix
    stride_ym,  # Stride for rows in output matrix
    stride_yn,  # Stride for columns in output matrix
    BLOCK_M: tl.constexpr,  # Block size for rows
    BLOCK_N: tl.constexpr   # Block size for columns
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = num_pid_m * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * num_pid_m
    group_size = num_pid_in_group
    pid_m = first_pid_m + (pid % num_pid_m)
    pid_n = (pid % group_size) // num_pid_m

    # Compute the block of the matrix to process
    rm = pid_m * BLOCK_M
    rn = pid_n * BLOCK_N
    XM = min(M, rm + BLOCK_M)
    XN = min(N, rn + BLOCK_N)

    # Load the block of the matrix
    X_block = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for i in range(BLOCK_M):
        for j in range(BLOCK_N):
            if rm + i < M and rn + j < N:
                X_block[i, j] = tl.load(X_ptr + (rm + i) * stride_xm + (rn + j) * stride_xn)

    # Compute the max value in the block
    max_val = tl.max(X_block, 1)
    max_val = tl.reshape(max_val, (BLOCK_M, 1))

    # Subtract the max value for numerical stability
    X_block = X_block - max_val

    # Compute the exponentials
    exp_X_block = tl.exp(X_block)

    # Compute the sum of exponentials
    sum_exp = tl.sum(exp_X_block, 1)
    sum_exp = tl.reshape(sum_exp, (BLOCK_M, 1))

    # Compute the log softmax
    log_sum_exp = tl.log(sum_exp)
    Y_block = X_block - log_sum_exp

    # Store the result back to the output matrix
    for i in range(BLOCK_M):
        for j in range(BLOCK_N):
            if rm + i < M and rn + j < N:
                tl.store(Y_ptr + (rm + i) * stride_ym + (rn + j) * stride_yn, Y_block[i, j])

import torch
import triton
import triton.language as tl

class LogSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, dim, dtype=None):
        # Ensure the input is contiguous
        X = X.contiguous()
        M, N = X.shape
        K = 1  # Auxiliary dimension (not used in this kernel)
        Y = torch.empty_like(X, dtype=dtype if dtype is not None else X.dtype)

        # Define the grid and block sizes
        BLOCK_M = 16
        BLOCK_N = 16
        grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)

        # Launch the kernel
        log_softmax_kernel[grid](
            X, Y, M, N, K, X.stride(0), X.stride(1), Y.stride(0), Y.stride(1),
            BLOCK_M, BLOCK_N
        )

        # Save necessary data for backward computation
        ctx.save_for_backward(X, Y)
        ctx.dim = dim

        return Y

    @staticmethod
    def backward(ctx, grad_output):
        X, Y = ctx.saved_tensors
        dim = ctx.dim
        M, N = X.shape
        K = 1  # Auxiliary dimension (not used in this kernel)
        grad_input = torch.empty_like(X)

        # Define the grid and block sizes
        BLOCK_M = 16
        BLOCK_N = 16
        grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)

        # Launch the backward kernel
        log_softmax_backward_kernel[grid](
            grad_output, Y, grad_input, M, N, K, grad_output.stride(0), grad_output.stride(1),
            Y.stride(0), Y.stride(1), grad_input.stride(0), grad_input.stride(1),
            BLOCK_M, BLOCK_N
        )

        return grad_input, None, None

@triton.jit
def log_softmax_backward_kernel(
    grad_output_ptr,  # Pointer to gradient of the output
    Y_ptr,           # Pointer to output matrix (log softmax result)
    grad_input_ptr,  # Pointer to gradient of the input
    M,               # Number of rows
    N,               # Number of columns
    K,               # Auxiliary dimension (not used in this kernel)
    stride_gom,      # Stride for rows in grad_output
    stride_gon,      # Stride for columns in grad_output
    stride_ym,       # Stride for rows in Y
    stride_yn,       # Stride for columns in Y
    stride_gim,      # Stride for rows in grad_input
    stride_gin,      # Stride for columns in grad_input
    BLOCK_M: tl.constexpr,  # Block size for rows
    BLOCK_N: tl.constexpr   # Block size for columns
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = num_pid_m * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * num_pid_m
    group_size = num_pid_in_group
    pid_m = first_pid_m + (pid % num_pid_m)
    pid_n = (pid % group_size) // num_pid_m

    # Compute the block of the matrix to process
    rm = pid_m * BLOCK_M
    rn = pid_n * BLOCK_N
    XM = min(M, rm + BLOCK_M)
    XN = min(N, rn + BLOCK_N)

    # Load the block of the matrices
    grad_output_block = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    Y_block = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for i in range(BLOCK_M):
        for j in range(BLOCK_N):
            if rm + i < M and rn + j < N:
                grad_output_block[i, j] = tl.load(grad_output_ptr + (rm + i) * stride_gom + (rn + j) * stride_gon)
                Y_block[i, j] = tl.load(Y_ptr + (rm + i) * stride_ym + (rn + j) * stride_yn)

    # Compute the sum of the gradients
    sum_grad_output = tl.sum(grad_output_block, 1)
    sum_grad_output = tl.reshape(sum_grad_output, (BLOCK_M, 1))

    # Compute the gradient of the input
    grad_input_block = grad_output_block - tl.exp(Y_block) * sum_grad_output

    # Store the result back to the gradient of the input matrix
    for i in range(BLOCK_M):
        for j in range(BLOCK_N):
            if rm + i < M and rn + j < N:
                tl.store(grad_input_ptr + (rm + i) * stride_gim + (rn + j) * stride_gin, grad_input_block[i, j])

def log_softmax(input, dim, dtype=None):
    return LogSoftmax.apply(input, dim, dtype)

import torch

# Create a random input tensor
input = torch.randn(1024, 1024, device='cuda')

# Compute the log softmax
output = log_softmax(input, dim=1)

# Print the output
print(output)
