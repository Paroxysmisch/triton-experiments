import triton
import triton.language as tl

# Define tunable parameters
TILE_K = 32
TILE_N = 32
ONE_TILE_PER_CTA = False

@triton.jit
def softmax_kernel_non_inner(output_ptr, input_ptr, M, N, K, stride_m, stride_n, stride_k, ONE_TILE_PER_CTA):
    pid = tl.program_id(axis=0)
    grid_m = tl.cdiv(M, TILE_N)
    grid_n = tl.cdiv(N, TILE_K)
    row = pid // grid_n
    col = pid % grid_n
    m = row * TILE_N + tl.arange(0, TILE_N)
    n = col * TILE_K + tl.arange(0, TILE_K)
    offsets = m[:, None] * stride_m + n[None, :] * stride_n
    values = input_ptr[offsets]
    max_val = tl.max(values, axis=1, keepdim=True)
    exp_values = tl.exp(values - max_val)
    sum_exp = tl.sum(exp_values, axis=1, keepdim=True)
    output_ptr[offsets] = exp_values / sum_exp

@triton.jit
def softmax_kernel_inner(output_ptr, input_ptr, M, N, K, stride_m, stride_n, stride_k, ONE_TILE_PER_CTA):
    pid = tl.program_id(axis=0)
    grid_m = tl.cdiv(M, TILE_N)
    grid_n = tl.cdiv(N, TILE_K)
    row = pid // grid_n
    col = pid % grid_n
    m = row * TILE_N + tl.arange(0, TILE_N)
    n = col * TILE_K + tl.arange(0, TILE_K)
    offsets = m[:, None] * stride_m + n[None, :] * stride_n
    values = input_ptr[offsets]
    max_val = tl.max(values, axis=1, keepdim=True)
    exp_values = tl.exp(values - max_val)
    sum_exp = tl.sum(exp_values, axis=1, keepdim=True)
    output_ptr[offsets] = exp_values / sum_exp

@triton.jit
def softmax_backward_kernel_non_inner(in_grad_ptr, output_ptr, grad_ptr, M, N, K, stride_m, stride_n, stride_k, ONE_TILE_PER_CTA):
    pid = tl.program_id(axis=0)
    grid_m = tl.cdiv(M, TILE_N)
    grid_n = tl.cdiv(N, TILE_K)
    row = pid // grid_n
    col = pid % grid_n
    m = row * TILE_N + tl.arange(0, TILE_N)
    n = col * TILE_K + tl.arange(0, TILE_K)
    offsets = m[:, None] * stride_m + n[None, :] * stride_n
    values = output_ptr[offsets]
    max_val = tl.max(values, axis=1, keepdim=True)
    exp_values = tl.exp(values - max_val)
    sum_exp = tl.sum(exp_values, axis=1, keepdim=True)
    grad_values = grad_ptr[offsets]
    in_grad_ptr[offsets] = grad_values * exp_values / sum_exp

@triton.jit
def softmax_backward_kernel_inner(in_grad_ptr, output_ptr, grad_ptr, M, N, K, stride_m, stride_n, stride_k, ONE_TILE_PER_CTA):
    pid = tl.program_id(axis=0)
    grid_m = tl.cdiv(M, TILE_N)
    grid_n = tl.cdiv(N, TILE_K)
    row = pid // grid_n
    col = pid % grid_n
    m = row * TILE_N + tl.arange(0, TILE_N)
    n = col * TILE_K + tl.arange(0, TILE_K)
    offsets = m[:, None] * stride_m + n[None, :] * stride_n
    values = output_ptr[offsets]
    max_val = tl.max(values, axis=1, keepdim=True)
    exp_values = tl.exp(values - max_val)
    sum_exp = tl.sum(exp_values, axis=1, keepdim=True)
    grad_values = grad_ptr[offsets]
    in_grad_ptr[offsets] = grad_values * exp_values / sum_exp

@triton.jit
def heur_tile_k(M, N, K):
    return TILE_K

@triton.jit
def heur_tile_n_non_inner(M, N, K):
    return TILE_N

# Softmax class with custom autograd function
class Softmax:
    @staticmethod
    @triton.autotune(
        configs=[
            triton.Config({'TILE_K': 32, 'TILE_N': 32}, num_stages=1, num_warps=4),
            triton.Config({'TILE_K': 64, 'TILE_N': 64}, num_stages=1, num_warps=4),
        ],
        key=['M', 'N', 'K']
    )
    def forward(input_ptr, output_ptr, M, N, K, stride_m, stride_n, stride_k, ONE_TILE_PER_CTA):
        if K > 1:
            softmax_kernel_non_inner[grid=triton.cdiv(M * N, TILE_N * TILE_K), block=(TILE_N, TILE_K, 1)](
                output_ptr, input_ptr, M, N, K, stride_m, stride_n, stride_k, ONE_TILE_PER_CTA
            )
        else:
            softmax_kernel_inner[grid=triton.cdiv(M * N, TILE_N * TILE_K), block=(TILE_N, TILE_K, 1)](
                output_ptr, input_ptr, M, N, K, stride_m, stride_n, stride_k, ONE_TILE_PER_CTA
            )

    @staticmethod
    @triton.autotune(
        configs=[
            triton.Config({'TILE_K': 32, 'TILE_N': 32}, num_stages=1, num_warps=4),
            triton.Config({'TILE_K': 64, 'TILE_N': 64}, num_stages=1, num_warps=4),
        ],
        key=['M', 'N', 'K']
    )
    def backward(in_grad_ptr, output_ptr, grad_ptr, M, N, K, stride_m, stride_n, stride_k, ONE_TILE_PER_CTA):
        if K > 1:
            softmax_backward_kernel_non_inner[grid=triton.cdiv(M * N, TILE_N * TILE_K), block=(TILE_N, TILE_K, 1)](
                in_grad_ptr, output_ptr, grad_ptr, M, N, K, stride_m, stride_n, stride_k, ONE_TILE_PER_CTA
            )
        else:
            softmax_backward_kernel_inner[grid=triton.cdiv(M * N, TILE_N * TILE_K), block=(TILE_N, TILE_K, 1)](
                in_grad_ptr, output_ptr, grad_ptr, M, N, K, stride_m, stride_n, stride_k, ONE_TILE_PER_CTA
            )
