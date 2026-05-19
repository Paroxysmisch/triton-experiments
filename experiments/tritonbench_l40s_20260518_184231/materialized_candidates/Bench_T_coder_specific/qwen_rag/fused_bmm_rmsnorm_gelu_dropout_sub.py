import torch
import triton
import triton.language as tl

# Constants for GELU approximation
SQRT_TWO_OVER_PI = 0.7978845608028654
ONE_MINUS_SQRT_TWO_OVER_PI = 0.2021154391971346

@triton.jit
def bmm_kernel(out_ptr, x_ptr, y_ptr, x_row_stride, y_col_stride, m, n, k, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    i = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    j = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for kk in range(0, k.size(0), BLOCK_SIZE_K):
        xk = tl.load(x_ptr + i[:, None] * x_row_stride + kk * k.size(1), mask=i[:, None] < m, other=0.0)
        yk = tl.load(y_ptr + kk * k.size(1) + j, mask=j < n, other=0.0)
        acc += xk * yk

    o_ptr = out_ptr + pid_m * BLOCK_SIZE_M * n + pid_n * BLOCK_SIZE_N
    tl.store(o_ptr + i * n + j, acc, mask=(i[:, None] < m) & (j < n))

@triton.jit
def rms_norm_kernel(out_ptr, x_ptr, norm_factor_ptr, stride, n, eps, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    row = pid * BLOCK_SIZE
    col = tl.arange(0, BLOCK_SIZE)

    x = tl.load(x_ptr + row * stride + col, mask=col < n, other=0.0)
    norm_factor = tl.load(norm_factor_ptr + row, mask=True, other=1.0)

    mean_square = tl.sum(x * x) / n
    norm = tl.sqrt(mean_square + eps)
    norm_factor[pid] = norm

    x_norm = x / norm
    out = x_norm * norm_factor[pid]

    tl.store(out_ptr + row * stride + col, out, mask=col < n)

@triton.jit
def gelu_kernel(out_ptr, x_ptr, stride, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    row = pid * BLOCK_SIZE
    col = tl.arange(0, BLOCK_SIZE)

    x = tl.load(x_ptr + row * stride + col, mask=col < n, other=0.0)
    g = 0.5 * (1.0 + tl.tanh(SQRT_TWO_OVER_PI * (x + 0.044715 * x * x * x)))
    out = x * g

    tl.store(out_ptr + row * stride + col, out, mask=col < n)

@triton.jit
def dropout_kernel(out_ptr, x_ptr, mask_ptr, stride, n, dropout_p, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    row = pid * BLOCK_SIZE
    col = tl.arange(0, BLOCK_SIZE)

    x = tl.load(x_ptr + row * stride + col, mask=col < n, other=0.0)
    keep_prob = 1.0 - dropout_p
    random_mask = tl.random.rand(BLOCK_SIZE) < keep_prob
    mask_ptr[row] = random_mask

    out = x * random_mask.to(x.dtype)
    tl.store(out_ptr + row * stride + col, out, mask=col < n)

@triton.jit
def fused_bmm_rmsnorm_gelu_dropout_sub(
    output_ptr, x_ptr, y_ptr, z_ptr, n, m, p, nrm_shape, dropout_p, training, approx, eps, stride_x, stride_y, stride_z, stride_out, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, BLOCK_SIZE_NRM: tl.constexpr, BLOCK_SIZE_GELU: tl.constexpr, BLOCK_SIZE_DROPOUT: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    i = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    j = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, m, BLOCK_SIZE_K):
        xk = tl.load(x_ptr + i[:, None] * stride_x + k * m, mask=i[:, None] < n, other=0.0)
        yk = tl.load(y_ptr + k * m + j, mask=j < p, other=0.0)
        acc += xk * yk

    out_ptr_norm = output_ptr + pid_m * BLOCK_SIZE_M * p + pid_n * BLOCK_SIZE_N
    norm_factor = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    rms_norm_kernel[BLANK, BLOCK_SIZE_NRM](output_ptr_norm, acc, norm_factor, stride_out, p, eps, BLOCK_SIZE_NRM)

    gelu_input = output_ptr_norm
    gelu_output = output_ptr_norm
    gelu_kernel[BLANK, BLOCK_SIZE_GELU](gelu_output, gelu_input, stride_out, p, BLOCK_SIZE_GELU)

    dropout_mask = tl.zeros((BLOCK_SIZE_M,), dtype=tl.bool)
    dropout_output = output_ptr_norm
    dropout_kernel[BLANK, BLOCK_SIZE_DROPOUT](dropout_output, gelu_output, dropout_mask, stride_out, p, dropout_p, BLOCK_SIZE_DROPOUT)

    tl.store(output_ptr_norm + i * p + j, dropout_output[i, j] - tl.load(z_ptr + i * p + j, mask=(i[:, None] < n) & (j < p)), mask=(i[:, None] < n) & (j < p))

def fused_bmm_rmsnorm_gelu_dropout_sub_torch(input1, input2, other, normalized_shape, dropout_p=0.5, training=True, approximate='none', eps=1e-5, *, out=None):
    assert input1.dim() == 3 and input2.dim() == 3 and other.dim() == 3
    assert input1.shape[0] == input2.shape[0] == other.shape[0]
    assert input1.shape[2] == input2.shape[1]
    assert input1.shape[1] == other.shape[1]
    assert input2.shape[2] == normalized_shape

    B, N, M = input1.shape
    _, M, P = input2.shape
    _, N, P = other.shape

    output = torch.empty((B, N, P), device=input1.device, dtype=input1.dtype)

    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 32
    BLOCK_SIZE_NRM = 32
    BLOCK_SIZE_GELU = 32
    BLOCK_SIZE_DROPOUT = 32

    fused_bmm_rmsnorm_gelu_dropout_sub[BLANK, BLOCK_SIZE_M, BLOCK_SIZE_N](
        output.data_ptr(),
        input1.data_ptr(),
        input2.data_ptr(),
        other.data_ptr(),
        N,
        M,
        P,
        normalized_shape,
        dropout_p,
        training,
        approximate,
        eps,
        input1.stride(0),
        input2.stride(0),
        other.stride(0),
        output.stride(0),
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
        BLOCK_SIZE_NRM,
        BLOCK_SIZE_GELU,
        BLOCK_SIZE_DROPOUT,
    )

    return output
