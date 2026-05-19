import triton
import triton.language as tl
import torch

# Helper function to determine optimal tile sizes and warp count
def heur_optimal_tile_sizes(M, N, K):
    TILE_N = 128 if N > 128 else N
    TILE_K = 32 if K > 32 else K
    num_warps = 4 if N > 128 else 1
    return TILE_N, TILE_K, num_warps

# Kernel for softmax on non-inner dimensions
@triton.jit
def softmax_kernel_non_inner(output_ptr, input_ptr, M, N, K, TILE_N: tl.constexpr, TILE_K: tl.constexpr, num_warps: tl.constexpr):
    pid = tl.program_id(0)
    # Compute start and end indices for this program
    row_start = pid * TILE_N
    row_end = min(row_start + TILE_N, N)
    # Initialize pointers for input and output
    input_ptrs = input_ptr + row_start
    output_ptrs = output_ptr + row_start
    # Load data
    data = tl.load(input_ptrs)
    # Compute softmax
    max_val = tl.max(data, axis=0)
    data = data - max_val
    exp_data = tl.exp(data)
    sum_exp = tl.sum(exp_data, axis=0)
    softmax_result = exp_data / sum_exp
    # Store results
    tl.store(output_ptrs, softmax_result)

# Kernel for softmax on inner dimensions
@triton.jit
def softmax_kernel_inner(output_ptr, input_ptr, M, N, TILE_N: tl.constexpr, num_warps: tl.constexpr):
    pid = tl.program_id(0)
    # Compute start and end indices for this program
    col_start = pid * TILE_N
    col_end = min(col_start + TILE_N, N)
    # Initialize pointers for input and output
    input_ptrs = input_ptr + col_start
    output_ptrs = output_ptr + col_start
    # Load data
    data = tl.load(input_ptrs)
    # Compute softmax
    max_val = tl.max(data, axis=1)
    data = data - max_val
    exp_data = tl.exp(data)
    sum_exp = tl.sum(exp_data, axis=1)
    softmax_result = exp_data / sum_exp
    # Store results
    tl.store(output_ptrs, softmax_result)

# Kernel for softmax backward pass on non-inner dimensions
@triton.jit
def softmax_backward_kernel_non_inner(grad_output_ptr, grad_input_ptr, softmax_output_ptr, M, N, K, TILE_N: tl.constexpr, TILE_K: tl.constexpr, num_warps: tl.constexpr):
    pid = tl.program_id(0)
    # Compute start and end indices for this program
    row_start = pid * TILE_N
    row_end = min(row_start + TILE_N, N)
    # Initialize pointers
    grad_output_ptrs = grad_output_ptr + row_start
    grad_input_ptrs = grad_input_ptr + row_start
    softmax_output_ptrs = softmax_output_ptr + row_start
    # Load data
    grad_output = tl.load(grad_output_ptrs)
    softmax_output = tl.load(softmax_output_ptrs)
    # Compute gradient
    dot_product = tl.sum(grad_output * softmax_output, axis=0)
    grad_input = softmax_output * (grad_output - dot_product)
    # Store results
    tl.store(grad_input_ptrs, grad_input)

# Kernel for softmax backward pass on inner dimensions
@triton.jit
def softmax_backward_kernel_inner(grad_output_ptr, grad_input_ptr, softmax_output_ptr, M, N, TILE_N: tl.constexpr, num_warps: tl.constexpr):
    pid = tl.program_id(0)
    # Compute start and end indices for this program
    col_start = pid * TILE_N
    col_end = min(col_start + TILE_N, N)
    # Initialize pointers
    grad_output_ptrs = grad_output_ptr + col_start
    grad_input_ptrs = grad_input_ptr + col_start
    softmax_output_ptrs = softmax_output_ptr + col_start
    # Load data
    grad_output = tl.load(grad_output_ptrs)
    softmax_output = tl.load(softmax_output_ptrs)
    # Compute gradient
    dot_product = tl.sum(grad_output * softmax_output, axis=1)
    grad_input = softmax_output * (grad_output - dot_product)
    # Store results
    tl.store(grad_input_ptrs, grad_input)

# PyTorch wrapper for the Triton kernels
class Softmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, inner=False):
        M, N = input.shape
        TILE_N, TILE_K, num_warps = heur_optimal_tile_sizes(M, N, N)
        output = torch.empty_like(input)
        
        if inner:
            softmax_kernel_inner[(M,)](output, input, M, N, TILE_N, num_warps)
        else:
            softmax_kernel_non_inner[(M,)](output, input, M, N, N, TILE_N, TILE_K, num_warps)
        
        ctx.save_for_backward(output)
        ctx.inner = inner
        return output

    @staticmethod
    def backward(ctx, grad_output):
        softmax_output, = ctx.saved_tensors
        M, N = grad_output.shape
        grad_input = torch.empty_like(grad_output)
        TILE_N, TILE_K, num_warps = heur_optimal_tile_sizes(M, N, N)
        
        if ctx.inner:
            softmax_backward_kernel_inner[(M,)](grad_output, grad_input, softmax_output, M, N, TILE_N, num_warps)
        else:
            softmax_backward_kernel_non_inner[(M,)](grad_output, grad_input, softmax_output, M, N, N, TILE_N, TILE_K, num_warps)
        
        return grad_input, None

# Usage
input_tensor = torch.randn(128, 128, device='cuda')
softmax_result = Softmax.apply(input_tensor, inner=False)
