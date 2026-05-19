import triton
import triton.language as tl

# Helper functions to determine optimal configurations
def heur_non_inner(M, N, K):
    TILE_N = 128
    TILE_K = 32
    return TILE_N, TILE_K

def heur_inner(M, N, K):
    ONE_TILE_PER_CTA = True
    return ONE_TILE_PER_CTA

# Forward Kernels
@triton.jit
def softmax_kernel_non_inner(output_ptr, input_ptr, M, N, K, TILE_N: tl.constexpr, TILE_K: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_tiles = (M + TILE_N - 1) // TILE_N
    tile_id = pid % num_tiles
    block_id = pid // num_tiles

    tile_start = tile_id * TILE_N
    tile_end = min(tile_start + TILE_N, M)

    offset = block_id * K
    row_start = tile_start * N
    row_end = tile_end * N

    for i in range(tile_start, tile_end):
        row = i * N
        max_val = -float('inf')
        for j in range(N):
            max_val = max(max_val, tl.load(input_ptr + row + j + offset))

        exp_sum = 0.0
        for j in range(N):
            exp_val = tl.exp(tl.load(input_ptr + row + j + offset) - max_val)
            exp_sum += exp_val
            tl.store(output_ptr + row + j + offset, exp_val)

        for j in range(N):
            tl.store(output_ptr + row + j + offset, tl.load(output_ptr + row + j + offset) / exp_sum)

@triton.jit
def softmax_kernel_inner(output_ptr, input_ptr, M, N, K, ONE_TILE_PER_CTA: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_warps = 4
    warp_id = pid % num_warps
    block_id = pid // num_warps

    tile_start = block_id * N
    tile_end = (block_id + 1) * N

    for i in range(M):
        row = i * N
        max_val = -float('inf')
        for j in range(tile_start, tile_end):
            max_val = max(max_val, tl.load(input_ptr + row + j))

        exp_sum = 0.0
        for j in range(tile_start, tile_end):
            exp_val = tl.exp(tl.load(input_ptr + row + j) - max_val)
            exp_sum += exp_val
            tl.store(output_ptr + row + j, exp_val)

        for j in range(tile_start, tile_end):
            tl.store(output_ptr + row + j, tl.load(output_ptr + row + j) / exp_sum)

# Backward Kernels
@triton.jit
def softmax_backward_kernel_non_inner(grad_input_ptr, grad_output_ptr, output_ptr, input_ptr, M, N, K, TILE_N: tl.constexpr, TILE_K: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_tiles = (M + TILE_N - 1) // TILE_N
    tile_id = pid % num_tiles
    block_id = pid // num_tiles

    tile_start = tile_id * TILE_N
    tile_end = min(tile_start + TILE_N, M)

    offset = block_id * K
    row_start = tile_start * N
    row_end = tile_end * N

    for i in range(tile_start, tile_end):
        row = i * N
        for j in range(N):
            output_val = tl.load(output_ptr + row + j + offset)
            grad_output_val = tl.load(grad_output_ptr + row + j + offset)
            grad_input_val = grad_output_val * output_val * (1 - output_val)
            tl.store(grad_input_ptr + row + j + offset, grad_input_val)

@triton.jit
def softmax_backward_kernel_inner(grad_input_ptr, grad_output_ptr, output_ptr, input_ptr, M, N, K, ONE_TILE_PER_CTA: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_warps = 4
    warp_id = pid % num_warps
    block_id = pid // num_warps

    tile_start = block_id * N
    tile_end = (block_id + 1) * N

    for i in range(M):
        row = i * N
        for j in range(tile_start, tile_end):
            output_val = tl.load(output_ptr + row + j)
            grad_output_val = tl.load(grad_output_ptr + row + j)
            grad_input_val = grad_output_val * output_val * (1 - output_val)
            tl.store(grad_input_ptr + row + j, grad_input_val)

### PyTorch Wrapper

import torch
from torch.autograd import Function

class Softmax(Function):
    @staticmethod
    def forward(ctx, input):
        M, N, K = input.shape
        output = torch.empty_like(input, device=input.device)
        TILE_N, TILE_K = heur_non_inner(M, N, K)
        ONE_TILE_PER_CTA = heur_inner(M, N, K)

        if K == 1:
            softmax_kernel_inner[lambda M, N, K: (M * N // 128,)](output, input, M, N, K, ONE_TILE_PER_CTA)
        else:
            softmax_kernel_non_inner[lambda M, N, K: (M * K // 128,)](output, input, M, N, K, TILE_N, TILE_K)

        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output, = ctx.saved_tensors
        grad_input = torch.empty_like(grad_output, device=grad_output.device)
        M, N, K = grad_output.shape
        TILE_N, TILE_K = heur_non_inner(M, N, K)
        ONE_TILE_PER_CTA = heur_inner(M, N, K)

        if K == 1:
            softmax_backward_kernel_inner[lambda M, N, K: (M * N // 128,)](grad_input, grad_output, output, grad_output, M, N, K, ONE_TILE_PER_CTA)
        else:
            softmax_backward_kernel_non_inner[lambda M, N, K: (M * K // 128,)](grad_input, grad_output, output, grad_output, M, N, K, TILE_N, TILE_K)

        return grad_input

# Example usage
if __name__ == "__main__":
    input_tensor = torch.randn(1024, 1024, 1, device='cuda')
    output_tensor = Softmax.apply(input_tensor)
    print(output_tensor)
