import torch
import triton
import triton.language as tl

@triton.jit
def log_softmax_kernel(
    input_ptr, output_ptr,
    M, N, K,
    stride_m, stride_n, stride_k,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid = tl.program_id(0)
    row_idx = pid // (N // BLOCK_N)
    col_idx = pid % (N // BLOCK_N)

    row_start = row_idx * BLOCK_M
    col_start = col_idx * BLOCK_N

    offsets_m = row_start + tl.arange(0, BLOCK_M)
    offsets_n = col_start + tl.arange(0, BLOCK_N)

    mask = (offsets_m[:, None] < M) & (offsets_n[None, :] < N)
    x = tl.load(input_ptr + offsets_m[:, None] * stride_m + offsets_n[None, :] * stride_n, mask=mask, other=-float('inf'))

    row_max = tl.max(x, axis=1)
    x = x - row_max[:, None]
    numerator = tl.exp(x)
    denominator = tl.sum(numerator, axis=1)
    log_denominator = tl.log(denominator)
    log_softmax = x - log_denominator[:, None]

    tl.store(output_ptr + offsets_m[:, None] * stride_m + offsets_n[None, :] * stride_n, log_softmax, mask=mask)

@triton.jit
def log_softmax_backward_kernel(
    grad_output_ptr, output_ptr, grad_input_ptr,
    M, N, K,
    stride_m, stride_n, stride_k,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid = tl.program_id(0)
    row_idx = pid // (N // BLOCK_N)
    col_idx = pid % (N // BLOCK_N)

    row_start = row_idx * BLOCK_M
    col_start = col_idx * BLOCK_N

    offsets_m = row_start + tl.arange(0, BLOCK_M)
    offsets_n = col_start + tl.arange(0, BLOCK_N)

    mask = (offsets_m[:, None] < M) & (offsets_n[None, :] < N)
    grad_output = tl.load(grad_output_ptr + offsets_m[:, None] * stride_m + offsets_n[None, :] * stride_n, mask=mask, other=0)
    output = tl.load(output_ptr + offsets_m[:, None] * stride_m + offsets_n[None, :] * stride_n, mask=mask, other=-float('inf'))

    grad_input = grad_output - tl.exp(output) * tl.sum(grad_output, axis=1)[:, None]
    tl.store(grad_input_ptr + offsets_m[:, None] * stride_m + offsets_n[None, :] * stride_n, grad_input, mask=mask)

class LogSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        M, N = input.shape
        output = torch.empty_like(input)
        BLOCK_M, BLOCK_N = 32, 128
        grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
        log_softmax_kernel[grid](
            input, output,
            M, N, 1,
            input.stride(0), input.stride(1), 0,
            BLOCK_M, BLOCK_N
        )
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output, = ctx.saved_tensors
        grad_input = torch.empty_like(grad_output)
        M, N = grad_output.shape
        BLOCK_M, BLOCK_N = 32, 128
        grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
        log_softmax_backward_kernel[grid](
            grad_output, output, grad_input,
            M, N, 1,
            grad_output.stride(0), grad_output.stride(1), 0,
            BLOCK_M, BLOCK_N
        )
        return grad_input

def log_softmax(input, dim=-1, dtype=None):
    if dim != -1 and dim != input.dim() - 1:
        input = input.transpose(dim, -1)
    if not input.is_contiguous():
        input = input.contiguous()
    output = LogSoftmax.apply(input)
    if dim != -1 and dim != input.dim() - 1:
        output = output.transpose(dim, -1)
    if dtype is not None:
        output = output.to(dtype)
    return output
