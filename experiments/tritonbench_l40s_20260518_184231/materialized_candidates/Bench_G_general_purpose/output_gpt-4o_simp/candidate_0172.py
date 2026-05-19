import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8),
    ],
    key=['M', 'N']
)
@triton.jit
def log_softmax_kernel(input_ptr, output_ptr, M, N, K, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    block_start_m = pid * BLOCK_M
    offsets_m = block_start_m + tl.arange(0, BLOCK_M)
    offsets_n = tl.arange(0, BLOCK_N)

    mask_m = offsets_m < M
    mask_n = offsets_n < N

    input_ptrs = input_ptr + offsets_m[:, None] * N + offsets_n[None, :]
    output_ptrs = output_ptr + offsets_m[:, None] * N + offsets_n[None, :]

    # Load input data
    input_data = tl.load(input_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=-float('inf'))

    # Compute max for numerical stability
    row_max = tl.max(input_data, axis=1, keepdim=True)
    input_data = input_data - row_max

    # Compute exp and sum
    exp_data = tl.exp(input_data)
    row_sum = tl.sum(exp_data, axis=1, keepdim=True)

    # Compute log softmax
    log_softmax_data = input_data - tl.log(row_sum)

    # Store result
    tl.store(output_ptrs, log_softmax_data, mask=mask_m[:, None] & mask_n[None, :])

import torch

class LogSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, dim):
        M, N = input.shape
        output = torch.empty_like(input)
        
        grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']),)
        log_softmax_kernel[grid](input, output, M, N, input.stride(0))

        ctx.save_for_backward(output)
        ctx.dim = dim
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output, = ctx.saved_tensors
        grad_input = torch.empty_like(grad_output)

        # Implement backward kernel call here (not provided)
        # log_softmax_backward_kernel(...)

        return grad_input, None

def log_softmax(input, dim=-1, dtype=None):
    if dtype is not None:
        input = input.to(dtype)
    return LogSoftmax.apply(input, dim)
