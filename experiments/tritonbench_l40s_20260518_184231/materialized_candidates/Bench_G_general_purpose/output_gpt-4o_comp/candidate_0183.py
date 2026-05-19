import triton
import triton.language as tl
import torch

# Kernel for softmax over non-inner dimensions
@triton.jit
def softmax_kernel_non_inner(output_ptr, input_ptr, M, N, K, TILE_K: tl.constexpr, TILE_N: tl.constexpr, ONE_TILE_PER_CTA: tl.constexpr):
    # Compute tile indices
    pid_k = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute start of the tile
    start_k = pid_k * TILE_K
    start_n = pid_n * TILE_N

    # Initialize pointers for the input and output
    input_tile_ptr = input_ptr + start_k * N + start_n
    output_tile_ptr = output_ptr + start_k * N + start_n

    # Load input tile
    input_tile = tl.load(input_tile_ptr, mask=(start_k < K) & (start_n < N), other=-float('inf'))

    # Compute softmax
    max_input = tl.max(input_tile, axis=0)
    input_tile = input_tile - max_input
    exp_input = tl.exp(input_tile)
    sum_exp_input = tl.sum(exp_input, axis=0)
    softmax_output = exp_input / sum_exp_input

    # Store the result
    tl.store(output_tile_ptr, softmax_output, mask=(start_k < K) & (start_n < N))

# Kernel for softmax over inner dimensions
@triton.jit
def softmax_kernel_inner(output_ptr, input_ptr, M, N, K, TILE_N: tl.constexpr):
    pid_n = tl.program_id(0)
    start_n = pid_n * TILE_N

    input_tile_ptr = input_ptr + start_n
    output_tile_ptr = output_ptr + start_n

    input_tile = tl.load(input_tile_ptr, mask=(start_n < N), other=-float('inf'))

    max_input = tl.max(input_tile, axis=0)
    input_tile = input_tile - max_input
    exp_input = tl.exp(input_tile)
    sum_exp_input = tl.sum(exp_input, axis=0)
    softmax_output = exp_input / sum_exp_input

    tl.store(output_tile_ptr, softmax_output, mask=(start_n < N))

# Backward kernel for non-inner dimensions
@triton.jit
def softmax_backward_kernel_non_inner(in_grad_ptr, out_grad_ptr, output_ptr, M, N, K, TILE_K: tl.constexpr, TILE_N: tl.constexpr):
    pid_k = tl.program_id(0)
    pid_n = tl.program_id(1)

    start_k = pid_k * TILE_K
    start_n = pid_n * TILE_N

    out_grad_tile_ptr = out_grad_ptr + start_k * N + start_n
    output_tile_ptr = output_ptr + start_k * N + start_n
    in_grad_tile_ptr = in_grad_ptr + start_k * N + start_n

    out_grad_tile = tl.load(out_grad_tile_ptr, mask=(start_k < K) & (start_n < N))
    output_tile = tl.load(output_tile_ptr, mask=(start_k < K) & (start_n < N))

    dot_product = tl.sum(out_grad_tile * output_tile, axis=0)
    in_grad_tile = output_tile * (out_grad_tile - dot_product)

    tl.store(in_grad_tile_ptr, in_grad_tile, mask=(start_k < K) & (start_n < N))

# Backward kernel for inner dimensions
@triton.jit
def softmax_backward_kernel_inner(in_grad_ptr, out_grad_ptr, output_ptr, M, N, K, TILE_N: tl.constexpr):
    pid_n = tl.program_id(0)
    start_n = pid_n * TILE_N

    out_grad_tile_ptr = out_grad_ptr + start_n
    output_tile_ptr = output_ptr + start_n
    in_grad_tile_ptr = in_grad_ptr + start_n

    out_grad_tile = tl.load(out_grad_tile_ptr, mask=(start_n < N))
    output_tile = tl.load(output_tile_ptr, mask=(start_n < N))

    dot_product = tl.sum(out_grad_tile * output_tile, axis=0)
    in_grad_tile = output_tile * (out_grad_tile - dot_product)

    tl.store(in_grad_tile_ptr, in_grad_tile, mask=(start_n < N))

# Heuristic functions for optimal tile sizes
def heur_tile_k(M, N, K):
    return min(32, K)

def heur_tile_n_non_inner(M, N, K):
    return min(128, N)

def heur_tile_n_inner(M, N, K):
    return min(128, N)

# Softmax class for PyTorch integration
class Softmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, dim):
        M, N, K = input.shape
        output = torch.empty_like(input)

        TILE_K = heur_tile_k(M, N, K)
        TILE_N = heur_tile_n_non_inner(M, N, K) if dim != -1 else heur_tile_n_inner(M, N, K)

        grid = lambda META: (triton.cdiv(K, META['TILE_K']), triton.cdiv(N, META['TILE_N']))

        if dim != -1:
            softmax_kernel_non_inner[grid](
                output, input, M, N, K,
                TILE_K=TILE_K, TILE_N=TILE_N,
                ONE_TILE_PER_CTA=False
            )
        else:
            softmax_kernel_inner[grid](
                output, input, M, N, K,
                TILE_N=TILE_N
            )

        ctx.save_for_backward(output)
        ctx.dim = dim
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output, = ctx.saved_tensors
        M, N, K = grad_output.shape
        grad_input = torch.empty_like(grad_output)

        TILE_K = heur_tile_k(M, N, K)
        TILE_N = heur_tile_n_non_inner(M, N, K) if ctx.dim != -1 else heur_tile_n_inner(M, N, K)

        grid = lambda META: (triton.cdiv(K, META['TILE_K']), triton.cdiv(N, META['TILE_N']))

        if ctx.dim != -1:
            softmax_backward_kernel_non_inner[grid](
                grad_input, grad_output, output, M, N, K,
                TILE_K=TILE_K, TILE_N=TILE_N
            )
        else:
            softmax_backward_kernel_inner[grid](
                grad_input, grad_output, output, M, N, K,
                TILE_N=TILE_N
            )

        return grad_input, None

# Example usage
input_tensor = torch.randn(128, 128, device='cuda')
output_tensor = Softmax.apply(input_tensor, -1)
