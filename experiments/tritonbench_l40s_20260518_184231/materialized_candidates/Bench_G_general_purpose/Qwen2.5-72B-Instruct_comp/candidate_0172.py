import triton
import triton.language as tl

@triton.jit
def log_softmax_kernel(X, Y, M, N, stride_xm, stride_xn, stride_ym, stride_yn, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_M

    offsets_m = block_start + tl.arange(0, BLOCK_M)
    offsets_n = tl.arange(0, BLOCK_N)
    X_block_ptr = X + (offsets_m[:, None] * stride_xm + offsets_n[None, :] * stride_xn)

    # Load the data into shared memory
    X_block = tl.load(X_block_ptr, mask=(offsets_m[:, None] < M) & (offsets_n[None, :] < N), other=-float('inf'))

    # Compute the maximum value in each row
    max_vals = tl.max(X_block, axis=1)
    max_vals = tl.where(offsets_m < M, max_vals, -float('inf'))

    # Subtract the maximum value from each element
    X_block = X_block - max_vals[:, None]

    # Compute the exponentials
    exp_X_block = tl.exp(X_block)

    # Compute the sum of exponentials in each row
    sum_exp = tl.sum(exp_X_block, axis=1)
    sum_exp = tl.where(offsets_m < M, sum_exp, 0.0)

    # Compute the log softmax
    log_softmax_block = X_block - tl.log(sum_exp[:, None])

    # Store the result back to global memory
    tl.store(Y + (offsets_m[:, None] * stride_ym + offsets_n[None, :] * stride_yn), log_softmax_block, mask=(offsets_m[:, None] < M) & (offsets_n[None, :] < N))

@triton.jit
def log_softmax_backward_kernel(dY, Y, dX, M, N, stride_dy_m, stride_dy_n, stride_y_m, stride_y_n, stride_dx_m, stride_dx_n, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_M

    offsets_m = block_start + tl.arange(0, BLOCK_M)
    offsets_n = tl.arange(0, BLOCK_N)
    dY_block_ptr = dY + (offsets_m[:, None] * stride_dy_m + offsets_n[None, :] * stride_dy_n)
    Y_block_ptr = Y + (offsets_m[:, None] * stride_y_m + offsets_n[None, :] * stride_y_n)
    dX_block_ptr = dX + (offsets_m[:, None] * stride_dx_m + offsets_n[None, :] * stride_dx_n)

    # Load the data into shared memory
    dY_block = tl.load(dY_block_ptr, mask=(offsets_m[:, None] < M) & (offsets_n[None, :] < N), other=0.0)
    Y_block = tl.load(Y_block_ptr, mask=(offsets_m[:, None] < M) & (offsets_n[None, :] < N), other=0.0)

    # Compute the sum of gradients in each row
    sum_dY = tl.sum(dY_block, axis=1)
    sum_dY = tl.where(offsets_m < M, sum_dY, 0.0)

    # Compute the gradient of the input
    dX_block = dY_block - tl.exp(Y_block) * sum_dY[:, None]

    # Store the result back to global memory
    tl.store(dX_block_ptr, dX_block, mask=(offsets_m[:, None] < M) & (offsets_n[None, :] < N))

import torch
from torch.autograd import Function

class LogSoftmax(Function):
    @staticmethod
    def forward(ctx, X, dim):
        # Ensure the input is contiguous
        X = X.contiguous()
        M, N = X.shape[dim], X.shape[1 - dim]
        Y = torch.empty_like(X)

        # Define grid and block sizes
        grid = (triton.cdiv(M, 128),)
        block = (128, 128)

        # Launch the Triton kernel
        log_softmax_kernel[grid, block](X, Y, M, N, X.stride(dim), X.stride(1 - dim), Y.stride(dim), Y.stride(1 - dim), BLOCK_M=128, BLOCK_N=128)

        # Save the output for the backward pass
        ctx.save_for_backward(Y)
        ctx.dim = dim

        return Y

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve the saved output
        Y, = ctx.saved_tensors
        dim = ctx.dim

        # Ensure the gradient is contiguous
        grad_output = grad_output.contiguous()
        grad_input = torch.empty_like(grad_output)

        # Define grid and block sizes
        grid = (triton.cdiv(grad_output.shape[dim], 128),)
        block = (128, 128)

        # Launch the Triton kernel
        log_softmax_backward_kernel[grid, block](grad_output, Y, grad_input, grad_output.shape[dim], grad_output.shape[1 - dim], grad_output.stride(dim), grad_output.stride(1 - dim), Y.stride(dim), Y.stride(1 - dim), grad_input.stride(dim), grad_input.stride(1 - dim), BLOCK_M=128, BLOCK_N=128)

        return grad_input, None

def log_softmax(X, dim, dtype=None):
    if dtype is not None:
        X = X.to(dtype)
    return LogSoftmax.apply(X, dim)

import torch

# Create a tensor
X = torch.randn(32, 64, requires_grad=True, device='cuda')

# Apply the log softmax
Y = log_softmax(X, dim=1)

# Compute the loss (for example, using cross-entropy)
target = torch.randint(0, 64, (32,), device='cuda')
loss = torch.nn.functional.nll_loss(Y, target)

# Backward pass
loss.backward()

# Print the gradient
print(X.grad)
