import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel_non_inner(output_ptr, input_ptr, M, N, K, TILE_M: tl.constexpr, TILE_N: tl.constexpr):
    row_idx = tl.program_id(0)
    col_idx = tl.program_id(1)
    
    input_row_start = input_ptr + row_idx * K
    output_row_start = output_ptr + row_idx * K

    col_offsets = tl.arange(0, TILE_N)
    row_offsets = tl.arange(0, TILE_M)
    
    input_ptrs = input_row_start + col_offsets
    row = tl.load(input_ptrs, mask=col_offsets < N, other=-float('inf'))
    
    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    
    output_ptrs = output_row_start + col_offsets
    tl.store(output_ptrs, softmax_output, mask=col_offsets < N)

@triton.jit
def softmax_kernel_inner(output_ptr, input_ptr, M, N, K, TILE_K: tl.constexpr):
    row_idx = tl.program_id(0)
    
    input_row_start = input_ptr + row_idx * K
    output_row_start = output_ptr + row_idx * K

    col_offsets = tl.arange(0, TILE_K)
    input_ptrs = input_row_start + col_offsets
    row = tl.load(input_ptrs, mask=col_offsets < K, other=-float('inf'))
    
    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    
    output_ptrs = output_row_start + col_offsets
    tl.store(output_ptrs, softmax_output, mask=col_offsets < K)

@triton.jit
def softmax_backward_kernel_non_inner(in_grad_ptr, out_grad_ptr, output_ptr, M, N, K, TILE_M: tl.constexpr, TILE_N: tl.constexpr):
    row_idx = tl.program_id(0)
    col_idx = tl.program_id(1)
    
    output_row_start = output_ptr + row_idx * K
    out_grad_row_start = out_grad_ptr + row_idx * K
    in_grad_row_start = in_grad_ptr + row_idx * K

    col_offsets = tl.arange(0, TILE_N)
    row_offsets = tl.arange(0, TILE_M)
    
    output_ptrs = output_row_start + col_offsets
    out_grad_ptrs = out_grad_row_start + col_offsets
    output = tl.load(output_ptrs, mask=col_offsets < N)
    out_grad = tl.load(out_grad_ptrs, mask=col_offsets < N)
    
    dot = tl.sum(out_grad * output, axis=0)
    in_grad = output * (out_grad - dot)
    
    in_grad_ptrs = in_grad_row_start + col_offsets
    tl.store(in_grad_ptrs, in_grad, mask=col_offsets < N)

@triton.jit
def softmax_backward_kernel_inner(in_grad_ptr, out_grad_ptr, output_ptr, M, N, K, TILE_K: tl.constexpr):
    row_idx = tl.program_id(0)
    
    output_row_start = output_ptr + row_idx * K
    out_grad_row_start = out_grad_ptr + row_idx * K
    in_grad_row_start = in_grad_ptr + row_idx * K

    col_offsets = tl.arange(0, TILE_K)
    output_ptrs = output_row_start + col_offsets
    out_grad_ptrs = out_grad_row_start + col_offsets
    output = tl.load(output_ptrs, mask=col_offsets < K)
    out_grad = tl.load(out_grad_ptrs, mask=col_offsets < K)
    
    dot = tl.sum(out_grad * output, axis=0)
    in_grad = output * (out_grad - dot)
    
    in_grad_ptrs = in_grad_row_start + col_offsets
    tl.store(in_grad_ptrs, in_grad, mask=col_offsets < K)

def softmax(x):
    M, N, K = x.shape
    y = torch.empty_like(x)
    
    TILE_M = 128
    TILE_N = 128
    TILE_K = 128
    
    grid_non_inner = lambda meta: (triton.cdiv(M, TILE_M), triton.cdiv(N, TILE_N))
    grid_inner = lambda meta: (M,)
    
    if K > 1:
        softmax_kernel_non_inner[grid_non_inner](
            y, x, M, N, K, TILE_M=TILE_M, TILE_N=TILE_N
        )
    else:
        softmax_kernel_inner[grid_inner](
            y, x, M, N, K, TILE_K=TILE_K
        )
    
    return y

def softmax_backward(in_grad, out_grad, output):
    M, N, K = output.shape
    in_grad = torch.empty_like(output)
    
    TILE_M = 128
    TILE_N = 128
    TILE_K = 128
    
    grid_non_inner = lambda meta: (triton.cdiv(M, TILE_M), triton.cdiv(N, TILE_N))
    grid_inner = lambda meta: (M,)
    
    if K > 1:
        softmax_backward_kernel_non_inner[grid_non_inner](
            in_grad, out_grad, output, M, N, K, TILE_M=TILE_M, TILE_N=TILE_N
        )
    else:
        softmax_backward_kernel_inner[grid_inner](
            in_grad, out_grad, output, M, N, K, TILE_K=TILE_K
        )
    
    return in_grad

class Softmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        y = softmax(x)
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, out_grad):
        y, = ctx.saved_tensors
        in_grad = softmax_backward(out_grad, out_grad, y)
        return in_grad

def softmax_autograd(x):
    return Softmax.apply(x)
