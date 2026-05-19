import triton
import triton.language as tl
import torch

# Define the kernel for softmax over non-inner dimensions
@triton.jit
def softmax_kernel_non_inner(output_ptr, input_ptr, M, N, K, stride_im, stride_in, stride_ik, stride_om, stride_on, stride_ok, TILE_K: tl.constexpr, TILE_N: tl.constexpr, ONE_TILE_PER_CTA: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start_m = pid * TILE_K
    block_start_n = pid * TILE_N

    # Compute the range of indices for the current block
    range_m = tl.arange(0, TILE_K) + block_start_m
    range_n = tl.arange(0, TILE_N) + block_start_n

    # Load the input values into shared memory
    input_tile = tl.zeros((TILE_K, TILE_N), dtype=tl.float32)
    for k in range(K):
        input_tile += tl.load(input_ptr + range_m[:, None] * stride_im + range_n[None, :] * stride_in + k * stride_ik)

    # Compute the maximum value in the tile
    max_val = tl.max(input_tile, 1)
    input_tile -= max_val[:, None]

    # Compute the exponentials
    exp_tile = tl.exp(input_tile)

    # Compute the sum of exponentials
    sum_exp = tl.sum(exp_tile, 1)

    # Compute the softmax values
    output_tile = exp_tile / sum_exp[:, None]

    # Store the results back to global memory
    for k in range(K):
        tl.store(output_ptr + range_m[:, None] * stride_om + range_n[None, :] * stride_on + k * stride_ok, output_tile)

# Define the kernel for softmax over inner dimensions
@triton.jit
def softmax_kernel_inner(output_ptr, input_ptr, M, N, K, stride_im, stride_in, stride_ik, stride_om, stride_on, stride_ok, TILE_K: tl.constexpr, TILE_N: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start_m = pid * TILE_K
    block_start_n = pid * TILE_N

    # Compute the range of indices for the current block
    range_m = tl.arange(0, TILE_K) + block_start_m
    range_n = tl.arange(0, TILE_N) + block_start_n

    # Load the input values into shared memory
    input_tile = tl.load(input_ptr + range_m[:, None] * stride_im + range_n[None, :] * stride_in)

    # Compute the maximum value in the tile
    max_val = tl.max(input_tile, 1)
    input_tile -= max_val[:, None]

    # Compute the exponentials
    exp_tile = tl.exp(input_tile)

    # Compute the sum of exponentials
    sum_exp = tl.sum(exp_tile, 1)

    # Compute the softmax values
    output_tile = exp_tile / sum_exp[:, None]

    # Store the results back to global memory
    tl.store(output_ptr + range_m[:, None] * stride_om + range_n[None, :] * stride_on, output_tile)

# Define the kernel for backward pass over non-inner dimensions
@triton.jit
def softmax_backward_kernel_non_inner(in_grad_ptr, output_ptr, input_ptr, M, N, K, stride_im, stride_in, stride_ik, stride_om, stride_on, stride_ok, stride_gm, stride_gn, stride_gk, TILE_K: tl.constexpr, TILE_N: tl.constexpr, ONE_TILE_PER_CTA: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start_m = pid * TILE_K
    block_start_n = pid * TILE_N

    # Compute the range of indices for the current block
    range_m = tl.arange(0, TILE_K) + block_start_m
    range_n = tl.arange(0, TILE_N) + block_start_n

    # Load the input and output values into shared memory
    input_tile = tl.zeros((TILE_K, TILE_N), dtype=tl.float32)
    output_tile = tl.zeros((TILE_K, TILE_N), dtype=tl.float32)
    for k in range(K):
        input_tile += tl.load(input_ptr + range_m[:, None] * stride_im + range_n[None, :] * stride_in + k * stride_ik)
        output_tile += tl.load(output_ptr + range_m[:, None] * stride_om + range_n[None, :] * stride_on + k * stride_ok)

    # Compute the gradient
    grad_tile = tl.load(in_grad_ptr + range_m[:, None] * stride_gm + range_n[None, :] * stride_gn)
    grad_tile *= output_tile
    sum_grad = tl.sum(grad_tile, 1)
    grad_tile -= output_tile * sum_grad[:, None]

    # Store the results back to global memory
    for k in range(K):
        tl.store(in_grad_ptr + range_m[:, None] * stride_gm + range_n[None, :] * stride_gn + k * stride_gk, grad_tile)

# Define the kernel for backward pass over inner dimensions
@triton.jit
def softmax_backward_kernel_inner(in_grad_ptr, output_ptr, input_ptr, M, N, K, stride_im, stride_in, stride_ik, stride_om, stride_on, stride_ok, stride_gm, stride_gn, stride_gk, TILE_K: tl.constexpr, TILE_N: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start_m = pid * TILE_K
    block_start_n = pid * TILE_N

    # Compute the range of indices for the current block
    range_m = tl.arange(0, TILE_K) + block_start_m
    range_n = tl.arange(0, TILE_N) + block_start_n

    # Load the input and output values into shared memory
    input_tile = tl.load(input_ptr + range_m[:, None] * stride_im + range_n[None, :] * stride_in)
    output_tile = tl.load(output_ptr + range_m[:, None] * stride_om + range_n[None, :] * stride_on)

    # Compute the gradient
    grad_tile = tl.load(in_grad_ptr + range_m[:, None] * stride_gm + range_n[None, :] * stride_gn)
    grad_tile *= output_tile
    sum_grad = tl.sum(grad_tile, 1)
    grad_tile -= output_tile * sum_grad[:, None]

    # Store the results back to global memory
    tl.store(in_grad_ptr + range_m[:, None] * stride_gm + range_n[None, :] * stride_gn, grad_tile)

# Heuristic functions to determine optimal tile sizes
def heur_tile_k(M, N, K):
    return min(128, M)

def heur_tile_n_non_inner(M, N, K):
    return min(128, N)

def heur_tile_n_inner(M, N, K):
    return min(128, N)

# Softmax class for PyTorch integration
class Softmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        M, N, K = input.shape
        output = torch.empty_like(input)
        TILE_K = heur_tile_k(M, N, K)
        TILE_N = heur_tile_n_non_inner(M, N, K) if K > 1 else heur_tile_n_inner(M, N, K)
        ONE_TILE_PER_CTA = True

        if K > 1:
            grid = lambda meta: (triton.cdiv(M, meta['TILE_K']) * triton.cdiv(N, meta['TILE_N']),)
            softmax_kernel_non_inner[grid](output, input, M, N, K, input.stride(0), input.stride(1), input.stride(2), output.stride(0), output.stride(1), output.stride(2), TILE_K, TILE_N, ONE_TILE_PER_CTA)
        else:
            grid = lambda meta: (triton.cdiv(M, meta['TILE_K']) * triton.cdiv(N, meta['TILE_N']),)
            softmax_kernel_inner[grid](output, input, M, N, K, input.stride(0), input.stride(1), input.stride(2), output.stride(0), output.stride(1), output.stride(2), TILE_K, TILE_N)

        ctx.save_for_backward(output, input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output, input = ctx.saved_tensors
        M, N, K = input.shape
        grad_input = torch.empty_like(input)
        TILE_K = heur_tile_k(M, N, K)
        TILE_N = heur_tile_n_non_inner(M, N, K) if K > 1 else heur_tile_n_inner(M, N, K)
        ONE_TILE_PER_CTA = True

        if K > 1:
            grid = lambda meta: (triton.cdiv(M, meta['TILE_K']) * triton.cdiv(N, meta['TILE_N']),)
            softmax_backward_kernel_non_inner[grid](grad_input, output, input, M, N, K, input.stride(0),
