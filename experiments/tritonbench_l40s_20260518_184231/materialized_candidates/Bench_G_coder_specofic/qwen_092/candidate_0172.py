import torch
import triton
import triton.language as tl
from log_softmax_kernel import log_softmax_kernel
from log_softmax_backward_kernel import log_softmax_backward_kernel

class LogSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, dim):
        input = input.contiguous()
        M, N = input.shape
        grid = (triton.cdiv(M, 32), triton.cdiv(N, 32))
        output = torch.empty_like(input)
        max_values = torch.empty((M, 1), dtype=input.dtype, device=input.device)
        exp_values = torch.empty_like(input)
        sum_values = torch.empty((M, 1), dtype=input.dtype, device=input.device)

        log_softmax_kernel[grid](output, input, max_values, exp_values, sum_values, input.stride(0), input.stride(1), M, N, BLOCK_M=32, BLOCK_N=32, num_warps=4)

        ctx.save_for_backward(input, output, max_values, exp_values, sum_values, dim)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, output, max_values, exp_values, sum_values, dim = ctx.saved_tensors
        grad_input = torch.zeros_like(input)
        grid = (triton.cdiv(input.shape[0], 32), triton.cdiv(input.shape[1], 32))

        log_softmax_backward_kernel[grid](grad_input, grad_output, output, exp_values, sum_values, input.stride(0), input.stride(1), input.shape[0], input.shape[1], BLOCK_M=32, BLOCK_N=32, num_warps=4)

        return grad_input, None

def log_softmax(input, dim):
    return LogSoftmax.apply(input, dim)
