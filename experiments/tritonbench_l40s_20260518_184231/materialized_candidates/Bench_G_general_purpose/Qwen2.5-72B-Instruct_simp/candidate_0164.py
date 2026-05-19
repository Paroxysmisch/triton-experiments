import triton
import triton.language as tl

# Define activation functions
@triton.jit
def tanh(x):
    return tl.math.tanh(x)

@triton.jit
def relu(x):
    return tl.math.max(x, 0)

@triton.jit
def gelu(x):
    return 0.5 * x * (1 + tl.math.tanh(0.7978845608 * (x + 0.044715 * x * x * x)))

@triton.jit
def fast_gelu(x):
    return 0.5 * x * (1 + tl.math.sign(x) * tl.math.sqrt(1 + 0.2888 * x * x))

# Define the kernel function
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def kernel_fma(
    A, B, C, bias, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, act_type: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if bias is not None:
        bias_ptr = bias + offs_bn
        bias_vals = tl.load(bias_ptr)
        accumulator += bias_vals

    c_ptrs = C + (offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn)
    if act_type == 0:
        accumulator = tanh(accumulator)
    elif act_type == 1:
        accumulator = relu(accumulator)
    elif act_type == 2:
        accumulator = gelu(accumulator)
    elif act_type == 3:
        accumulator = fast_gelu(accumulator)

    tl.store(c_ptrs, accumulator.to(C.dtype.element_ty))

import torch
import torch.nn as nn
from torch.autograd import Function

class LinearLayerFunction(Function):
    @staticmethod
    def forward(ctx, A, B, bias, act_type):
        M, K = A.shape
        N = B.shape[1]
        C = torch.empty((M, N), device=A.device, dtype=A.dtype)
        grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
        kernel_fma[grid](A, B, C, bias, M, N, K, A.stride(0), A.stride(1), B.stride(0), B.stride(1), C.stride(0), C.stride(1), act_type)
        ctx.save_for_backward(A, B, bias)
        ctx.act_type = act_type
        return C

    @staticmethod
    def backward(ctx, grad_output):
        A, B, bias = ctx.saved_tensors
        act_type = ctx.act_type
        M, K = A.shape
        N = B.shape[1]

        # Compute gradient with respect to C
        grad_C = grad_output.clone()

        # Apply activation function gradient
        if act_type == 0:
            grad_C *= 1 - grad_C * grad_C
        elif act_type == 1:
            grad_C *= (A @ B > 0).float()
        elif act_type == 2:
            x = A @ B
            grad_C *= 0.5 * (1 + 0.7978845608 * (x + 0.044715 * x * x * x) * (1 - tl.math.tanh(0.7978845608 * (x + 0.044715 * x * x * x)) ** 2))
        elif act_type == 3:
            x = A @ B
            grad_C *= 0.5 * (1 + 0.7978845608 * x * (1 + 0.1444 * x * x))

        # Compute gradient with respect to A and B
        grad_A = grad_C @ B.T
        grad_B = A.T @ grad_C

        # Compute gradient with respect to bias
        if bias is not None:
            grad_bias = grad_C.sum(dim=0)
        else:
            grad_bias = None

        return grad_A, grad_B, grad_bias, None

class LinearLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, act_type=0):
        super(LinearLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features)) if bias else None
        self.act_type = act_type

    def forward(self, input):
        return LinearLayerFunction.apply(input, self.weight, self.bias, self.act_type)

# Wrapper function
def linear_layer(input, weight, bias=None, act_type=0):
    return LinearLayerFunction.apply(input, weight, bias, act_type)
