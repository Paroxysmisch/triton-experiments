import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['n_cols'],
)
@triton.jit
def log_softmax_kernel(
    input_ptr,
    output_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start + col_offsets
    mask = col_offsets < n_cols
    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
    
    row_max = tl.max(row, axis=0)
    row_minus_max = row - row_max
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    log_denominator = tl.log(denominator)
    log_softmax_row = row_minus_max - log_denominator
    
    output_row_start = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start + col_offsets
    tl.store(output_ptrs, log_softmax_row, mask=mask)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['n_cols'],
)
@triton.jit
def log_softmax_backward_kernel(
    grad_output_ptr,
    output_ptr,
    grad_input_ptr,
    grad_row_stride,
    output_row_stride,
    grad_input_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    grad_row_start = grad_output_ptr + row_idx * grad_row_stride
    output_row_start = output_ptr + row_idx * output_row_stride
    grad_input_row_start = grad_input_ptr + row_idx * grad_input_row_stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    grad_ptrs = grad_row_start + col_offsets
    output_ptrs = output_row_start + col_offsets
    grad_input_ptrs = grad_input_row_start + col_offsets

    grad_output = tl.load(grad_ptrs, mask=mask, other=0.0)
    output = tl.load(output_ptrs, mask=mask, other=-float('inf'))

    softmax = tl.exp(output)
    sum_grad = tl.sum(grad_output, axis=0)
    grad_input = grad_output - softmax * sum_grad

    tl.store(grad_input_ptrs, grad_input, mask=mask)

class LogSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dim):
        original_shape = x.shape
        if dim < 0:
            dim += x.dim()
        ctx.dim = dim
        
        x_contig = x.contiguous()
        x_2d = x_contig.view(-1, x_contig.size(dim))
        M, N = x_2d.shape
        
        output_2d = torch.empty_like(x_2d)
        
        log_softmax_kernel[(M,)](x_2d, output_2d, x_2d.stride(0), output_2d.stride(0), N)
        
        ctx.save_for_backward(output_2d)
        ctx.original_shape = original_shape
        return output_2d.view(original_shape)

    @staticmethod
    def backward(ctx, grad_output):
        output_2d, = ctx.saved_tensors
        original_shape = ctx.original_shape
        dim = ctx.dim
        
        grad_output_contig = grad_output.contiguous()
        grad_output_2d = grad_output_contig.view(-1, grad_output_contig.size(dim))
        M, N = grad_output_2d.shape
        
        grad_input_2d = torch.empty_like(grad_output_2d)
        
        log_softmax_backward_kernel[(M,)](
            grad_output_2d, output_2d, grad_input_2d,
            grad_output_2d.stride(0), output_2d.stride(0), grad_input_2d.stride(0), N
        )
        
        return grad_input_2d.view(original_shape), None

def log_softmax(x, dim=-1, dtype=None):
    if dtype is not None:
        x = x.to(dtype)
    if dim < 0:
        dim += x.dim()
    if not x.is_contiguous():
        x = x.contiguous()
    return LogSoftmax.apply(x, dim)
