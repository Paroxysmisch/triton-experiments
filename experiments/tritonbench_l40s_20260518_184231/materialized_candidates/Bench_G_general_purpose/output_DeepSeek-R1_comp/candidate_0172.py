import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 1, 'BLOCK_N': 128}, num_warps=4),
        triton.Config({'BLOCK_M': 1, 'BLOCK_N': 256}, num_warps=4),
        triton.Config({'BLOCK_M': 1, 'BLOCK_N': 512}, num_warps=4),
        triton.Config({'BLOCK_M': 1, 'BLOCK_N': 1024}, num_warps=4),
    ],
    key=['N'],
)
@triton.heuristics({
    'BLOCK_M': lambda args: 1,
    'BLOCK_N': lambda args: triton.next_power_of_2(args['N']),
})
@triton.jit
def log_softmax_kernel(
    output_ptr, input_ptr, input_row_stride, output_row_stride, M, N,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    row_idx = tl.program_id(0)
    if row_idx >= M:
        return

    input_row = input_ptr + row_idx * input_row_stride
    output_row = output_ptr + row_idx * output_row_stride

    # Compute row max
    row_max = -float('inf')
    for block_col in range(0, N, BLOCK_N):
        col_offsets = block_col + tl.arange(0, BLOCK_N)
        mask = col_offsets < N
        x = tl.load(input_row + col_offsets, mask=mask, other=-float('inf'))
        row_max = tl.maximum(row_max, tl.max(x, 0))

    # Compute sum exp
    sum_exp = 0.0
    for block_col in range(0, N, BLOCK_N):
        col_offsets = block_col + tl.arange(0, BLOCK_N)
        mask = col_offsets < N
        x = tl.load(input_row + col_offsets, mask=mask, other=0.0)
        x_shifted = tl.exp(x - row_max)
        sum_exp += tl.sum(x_shifted, 0)

    log_sum_exp = tl.log(sum_exp)

    # Compute output
    for block_col in range(0, N, BLOCK_N):
        col_offsets = block_col + tl.arange(0, BLOCK_N)
        mask = col_offsets < N
        x = tl.load(input_row + col_offsets, mask=mask)
        output = x - row_max - log_sum_exp
        tl.store(output_row + col_offsets, output, mask=mask)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 1, 'BLOCK_N': 128}, num_warps=4),
        triton.Config({'BLOCK_M': 1, 'BLOCK_N': 256}, num_warps=4),
        triton.Config({'BLOCK_M': 1, 'BLOCK_N': 512}, num_warps=4),
        triton.Config({'BLOCK_M': 1, 'BLOCK_N': 1024}, num_warps=4),
    ],
    key=['N'],
)
@triton.heuristics({
    'BLOCK_M': lambda args: 1,
    'BLOCK_N': lambda args: triton.next_power_of_2(args['N']),
})
@triton.jit
def log_softmax_backward_kernel(
    grad_input_ptr, grad_output_ptr, output_ptr,
    grad_output_row_stride, output_row_stride, grad_input_row_stride,
    M, N,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    row_idx = tl.program_id(0)
    if row_idx >= M:
        return

    grad_output_row = grad_output_ptr + row_idx * grad_output_row_stride
    output_row = output_ptr + row_idx * output_row_stride
    grad_input_row = grad_input_ptr + row_idx * grad_input_row_stride

    # Compute sum of grad_output
    sum_grad = 0.0
    for block_col in range(0, N, BLOCK_N):
        col_offsets = block_col + tl.arange(0, BLOCK_N)
        mask = col_offsets < N
        grad = tl.load(grad_output_row + col_offsets, mask=mask, other=0.0)
        sum_grad += tl.sum(grad, 0)

    # Compute gradients
    for block_col in range(0, N, BLOCK_N):
        col_offsets = block_col + tl.arange(0, BLOCK_N)
        mask = col_offsets < N
        grad = tl.load(grad_output_row + col_offsets, mask=mask)
        s = tl.load(output_row + col_offsets, mask=mask)
        exp_s = tl.exp(s)
        grad_elem = grad - exp_s * sum_grad
        tl.store(grad_input_row + col_offsets, grad_elem, mask=mask)

class LogSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dim):
        x_contig = x.contiguous()
        dim = dim if dim >= 0 else x.dim() + dim
        
        # Move target dim to last dimension and flatten
        x_permuted = x_contig.transpose(dim, -1)
        original_shape = x_permuted.size()
        x_2d = x_permuted.view(-1, original_shape[-1])
        M, N = x_2d.size(0), x_2d.size(1)
        
        # Allocate output tensor
        output_2d = torch.empty_like(x_2d)
        
        # Launch kernel
        grid = (M,)
        log_softmax_kernel[grid](
            output_2d, x_2d,
            x_2d.stride(0), output_2d.stride(0),
            M, N
        )
        
        # Reshape back to original dimensions
        output = output_2d.view(original_shape).transpose(dim, -1).contiguous()
        ctx.save_for_backward(output)
        ctx.dim = dim
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_output_contig = grad_output.contiguous()
        output, = ctx.saved_tensors
        dim = ctx.dim
        
        # Move target dim to last dimension and flatten
        output_permuted = output.transpose(dim, -1)
        grad_output_permuted = grad_output_contig.transpose(dim, -1)
        original_shape = output_permuted.size()
        grad_output_2d = grad_output_permuted.view(-1, original_shape[-1])
        output_2d = output_permuted.view(-1, original_shape[-1])
        M, N = grad_output_2d.size(0), grad_output_2d.size(1)
        
        # Allocate gradient input tensor
        grad_input_2d = torch.empty_like(grad_output_2d)
        
        # Launch backward kernel
        grid = (M,)
        log_softmax_backward_kernel[grid](
            grad_input_2d, grad_output_2d, output_2d,
            grad_output_2d.stride(0), output_2d.stride(0), grad_input_2d.stride(0),
            M, N
        )
        
        # Reshape back to original dimensions
        grad_input = grad_input_2d.view(original_shape).transpose(dim, -1).contiguous()
        return grad_input, None

def log_softmax(x, dim=-1, dtype=None):
    if dtype is not None:
        x = x.to(dtype)
    return LogSoftmax.apply(x.contiguous(), dim)

x = torch.randn(128, 256, device='cuda')
output = log_softmax(x, dim=1)
loss = output.sum()
loss.backward()
