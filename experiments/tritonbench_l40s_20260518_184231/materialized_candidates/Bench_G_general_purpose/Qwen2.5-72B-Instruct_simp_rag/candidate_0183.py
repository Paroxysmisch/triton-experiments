import torch
import triton
import triton.language as tl

# Helper function to determine optimal block size
def heur_(M, N, K):
    TILE_N = triton.next_power_of_2(N)
    TILE_K = triton.next_power_of_2(K)
    ONE_TILE_PER_CTA = (TILE_N * TILE_K) // (32 * 32)  # 32 threads per warp, 32 warps per CTA
    return TILE_N, TILE_K, ONE_TILE_PER_CTA

# Forward Kernel for Non-Inner Dimensions
@triton.jit
def softmax_kernel_non_inner(output_ptr, input_ptr, M, N, K, TILE_N, TILE_K, ONE_TILE_PER_CTA):
    pid = tl.program_id(0)
    block_id = pid // ONE_TILE_PER_CTA
    warp_id = pid % ONE_TILE_PER_CTA
    row_id = block_id * TILE_K + warp_id * 32
    col_id = block_id * TILE_N + tl.arange(0, TILE_N)

    mask = (row_id < M) & (col_id < N)
    input_ptrs = input_ptr + row_id * N + col_id
    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator

    output_ptrs = output_ptr + row_id * N + col_id
    tl.store(output_ptrs, softmax_output, mask=mask)

# Forward Kernel for Inner Dimensions
@triton.jit
def softmax_kernel_inner(output_ptr, input_ptr, M, N, K, TILE_N, TILE_K, ONE_TILE_PER_CTA):
    pid = tl.program_id(0)
    block_id = pid // ONE_TILE_PER_CTA
    warp_id = pid % ONE_TILE_PER_CTA
    row_id = block_id * TILE_K + warp_id * 32
    col_id = block_id * TILE_N + tl.arange(0, TILE_N)

    mask = (row_id < M) & (col_id < N)
    input_ptrs = input_ptr + row_id * N + col_id
    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator

    output_ptrs = output_ptr + row_id * N + col_id
    tl.store(output_ptrs, softmax_output, mask=mask)

# Backward Kernel for Non-Inner Dimensions
@triton.jit
def softmax_backward_kernel_non_inner(grad_output_ptr, output_ptr, input_ptr, grad_input_ptr, M, N, K, TILE_N, TILE_K, ONE_TILE_PER_CTA):
    pid = tl.program_id(0)
    block_id = pid // ONE_TILE_PER_CTA
    warp_id = pid % ONE_TILE_PER_CTA
    row_id = block_id * TILE_K + warp_id * 32
    col_id = block_id * TILE_N + tl.arange(0, TILE_N)

    mask = (row_id < M) & (col_id < N)
    grad_output_ptrs = grad_output_ptr + row_id * N + col_id
    output_ptrs = output_ptr + row_id * N + col_id
    grad_output = tl.load(grad_output_ptrs, mask=mask, other=0.0)
    output = tl.load(output_ptrs, mask=mask, other=0.0)

    grad_input = output * (grad_output - tl.sum(grad_output * output, axis=0))

    grad_input_ptrs = grad_input_ptr + row_id * N + col_id
    tl.store(grad_input_ptrs, grad_input, mask=mask)

# Backward Kernel for Inner Dimensions
@triton.jit
def softmax_backward_kernel_inner(grad_output_ptr, output_ptr, input_ptr, grad_input_ptr, M, N, K, TILE_N, TILE_K, ONE_TILE_PER_CTA):
    pid = tl.program_id(0)
    block_id = pid // ONE_TILE_PER_CTA
    warp_id = pid % ONE_TILE_PER_CTA
    row_id = block_id * TILE_K + warp_id * 32
    col_id = block_id * TILE_N + tl.arange(0, TILE_N)

    mask = (row_id < M) & (col_id < N)
    grad_output_ptrs = grad_output_ptr + row_id * N + col_id
    output_ptrs = output_ptr + row_id * N + col_id
    grad_output = tl.load(grad_output_ptrs, mask=mask, other=0.0)
    output = tl.load(output_ptrs, mask=mask, other=0.0)

    grad_input = output * (grad_output - tl.sum(grad_output * output, axis=0))

    grad_input_ptrs = grad_input_ptr + row_id * N + col_id
    tl.store(grad_input_ptrs, grad_input, mask=mask)

# Softmax Class
class Softmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        M, N = input.shape
        K = 1  # For simplicity, assuming K is 1 for row-wise softmax
        TILE_N, TILE_K, ONE_TILE_PER_CTA = heur_(M, N, K)

        output = torch.empty_like(input)
        if N > 1:
            softmax_kernel_non_inner[(M * ONE_TILE_PER_CTA,)](
                output, input, M, N, K, TILE_N, TILE_K, ONE_TILE_PER_CTA
            )
        else:
            softmax_kernel_inner[(M * ONE_TILE_PER_CTA,)](
                output, input, M, N, K, TILE_N, TILE_K, ONE_TILE_PER_CTA
            )

        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output, = ctx.saved_tensors
        M, N = output.shape
        K = 1  # For simplicity, assuming K is 1 for row-wise softmax
        TILE_N, TILE_K, ONE_TILE_PER_CTA = heur_(M, N, K)

        grad_input = torch.empty_like(output)
        if N > 1:
            softmax_backward_kernel_non_inner[(M * ONE_TILE_PER_CTA,)](
                grad_output, output, output, grad_input, M, N, K, TILE_N, TILE_K, ONE_TILE_PER_CTA
            )
        else:
            softmax_backward_kernel_inner[(M * ONE_TILE_PER_CTA,)](
                grad_output, output, output, grad_input, M, N, K, TILE_N, TILE_K, ONE_TILE_PER_CTA
            )

        return grad_input

# Example Usage
if __name__ == "__main__":
    x = torch.randn(1024, 1024, device='cuda')
    y = Softmax.apply(x)
    print(y)
