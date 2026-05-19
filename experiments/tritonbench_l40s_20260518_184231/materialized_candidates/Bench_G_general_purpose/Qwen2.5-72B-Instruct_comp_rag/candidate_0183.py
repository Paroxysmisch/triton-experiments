import torch
import triton
import triton.language as tl

# Define the forward kernel for non-inner dimensions
@triton.jit
def softmax_kernel_non_inner(output_ptr, input_ptr, M, N, K, TILE_K: tl.constexpr, TILE_N: tl.constexpr, ONE_TILE_PER_CTA: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * TILE_K * TILE_N
    offsets_m = tl.arange(0, TILE_K)[:, None]
    offsets_n = tl.arange(0, TILE_N)[None, :]
    input_offsets = block_start + offsets_m * N + offsets_n
    input_ptrs = input_ptr + input_offsets
    input_mask = (input_offsets < M * N)
    input_data = tl.load(input_ptrs, mask=input_mask, other=-float('inf'))
    
    max_val = tl.max(input_data, axis=1)[:, None]
    input_data -= max_val
    exp_data = tl.exp(input_data)
    sum_exp = tl.sum(exp_data, axis=1)[:, None]
    softmax_output = exp_data / sum_exp
    
    output_offsets = block_start + offsets_m * N + offsets_n
    output_ptrs = output_ptr + output_offsets
    tl.store(output_ptrs, softmax_output, mask=input_mask)

# Define the forward kernel for inner dimensions
@triton.jit
def softmax_kernel_inner(output_ptr, input_ptr, M, N, K, TILE_K: tl.constexpr, TILE_N: tl.constexpr, ONE_TILE_PER_CTA: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * TILE_K * K
    offsets_m = tl.arange(0, TILE_K)[:, None]
    offsets_k = tl.arange(0, K)[None, :]
    input_offsets = block_start + offsets_m * K + offsets_k
    input_ptrs = input_ptr + input_offsets
    input_mask = (input_offsets < M * K)
    input_data = tl.load(input_ptrs, mask=input_mask, other=-float('inf'))
    
    max_val = tl.max(input_data, axis=1)[:, None]
    input_data -= max_val
    exp_data = tl.exp(input_data)
    sum_exp = tl.sum(exp_data, axis=1)[:, None]
    softmax_output = exp_data / sum_exp
    
    output_offsets = block_start + offsets_m * K + offsets_k
    output_ptrs = output_ptr + output_offsets
    tl.store(output_ptrs, softmax_output, mask=input_mask)

# Define the backward kernel for non-inner dimensions
@triton.jit
def softmax_backward_kernel_non_inner(in_grad_ptr, output_ptr, input_ptr, M, N, K, TILE_K: tl.constexpr, TILE_N: tl.constexpr, ONE_TILE_PER_CTA: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * TILE_K * TILE_N
    offsets_m = tl.arange(0, TILE_K)[:, None]
    offsets_n = tl.arange(0, TILE_N)[None, :]
    input_offsets = block_start + offsets_m * N + offsets_n
    input_ptrs = input_ptr + input_offsets
    output_ptrs = output_ptr + input_offsets
    in_grad_ptrs = in_grad_ptr + input_offsets
    
    input_mask = (input_offsets < M * N)
    input_data = tl.load(input_ptrs, mask=input_mask, other=-float('inf'))
    output_data = tl.load(output_ptrs, mask=input_mask, other=0.0)
    in_grad_data = tl.load(in_grad_ptrs, mask=input_mask, other=0.0)
    
    grad_output = output_data * (in_grad_data - tl.sum(output_data * in_grad_data, axis=1)[:, None])
    
    tl.store(input_ptrs, grad_output, mask=input_mask)

# Define the backward kernel for inner dimensions
@triton.jit
def softmax_backward_kernel_inner(in_grad_ptr, output_ptr, input_ptr, M, N, K, TILE_K: tl.constexpr, TILE_N: tl.constexpr, ONE_TILE_PER_CTA: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * TILE_K * K
    offsets_m = tl.arange(0, TILE_K)[:, None]
    offsets_k = tl.arange(0, K)[None, :]
    input_offsets = block_start + offsets_m * K + offsets_k
    input_ptrs = input_ptr + input_offsets
    output_ptrs = output_ptr + input_offsets
    in_grad_ptrs = in_grad_ptr + input_offsets
    
    input_mask = (input_offsets < M * K)
    input_data = tl.load(input_ptrs, mask=input_mask, other=-float('inf'))
    output_data = tl.load(output_ptrs, mask=input_mask, other=0.0)
    in_grad_data = tl.load(in_grad_ptrs, mask=input_mask, other=0.0)
    
    grad_output = output_data * (in_grad_data - tl.sum(output_data * in_grad_data, axis=1)[:, None])
    
    tl.store(input_ptrs, grad_output, mask=input_mask)

# Heuristic functions to determine optimal tile sizes
def heur_tile_k(M, K):
    if K > 1024:
        return 256
    elif K > 512:
        return 128
    else:
        return 64

def heur_tile_n_non_inner(M, N):
    if N > 1024:
        return 256
    elif N > 512:
        return 128
    else:
        return 64

def heur_tile_n_inner(M, N):
    if N > 1024:
        return 256
    elif N > 512:
        return 128
    else:
        return 64

# Softmax class for PyTorch integration
class Softmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        M, N, K = input.shape
        output = torch.empty_like(input)
        
        if K > 1:
            TILE_K = heur_tile_k(M, K)
            TILE_N = heur_tile_n_non_inner(M, N)
            ONE_TILE_PER_CTA = False
            grid = lambda meta: (triton.cdiv(M * N, meta['TILE_K'] * meta['TILE_N']),)
            softmax_kernel_non_inner[grid](output, input, M, N, K, TILE_K, TILE_N, ONE_TILE_PER_CTA)
        else:
            TILE_K = heur_tile_k(M, K)
            TILE_N = heur_tile_n_inner(M, N)
            ONE_TILE_PER_CTA = False
            grid = lambda meta: (triton.cdiv(M * K, meta['TILE_K'] * meta['TILE_N']),)
            softmax_kernel_inner[grid](output, input, M, N, K, TILE_K, TILE_N, ONE_TILE_PER_CTA)
        
        ctx.save_for_backward(output, input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output, input = ctx.saved_tensors
        M, N, K = input.shape
        grad_input = torch.empty_like(input)
        
        if K > 1:
            TILE_K = heur_tile_k(M, K)
            TILE_N = heur_tile_n_non_inner(M, N)
            ONE_TILE_PER_CTA = False
            grid = lambda meta: (triton.cdiv(M * N, meta['TILE_K'] * meta['TILE_N']),)
            softmax_backward_kernel_non_inner[grid](grad_input, output, input, M, N, K, TILE_K, TILE_N, ONE_TILE_PER_CTA)
        else:
            TILE_K = heur_tile_k(M, K)
            TILE_N = heur_tile_n_inner(M, N)
            ONE_TILE_PER_CTA = False
            grid = lambda meta: (triton.cdiv(M * K, meta['TILE_K'] * meta['TILE_N']),)
            softmax_backward_kernel_inner[grid](grad_input, output, input, M, N, K, TILE_K, TILE_N, ONE_TILE_PER_CTA)
        
        return grad_input

# Example usage
if __name__ == "__main__":
    x = torch.randn(1024, 1024, 1, device='cuda')
    y = Softmax.apply(x)
    print(y)
