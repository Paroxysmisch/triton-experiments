import torch
import triton
import triton.language as tl

# Auxiliary functions for activations
@triton.jit
def tanh(x):
    return tl.tanh(x)

@triton.jit
def relu(x):
    return tl.where(x >= 0, x, 0)

@triton.jit
def gelu(x):
    return 0.5 * x * (1 + tl.erf(x / tl.sqrt(2.0)))

@triton.jit
def fast_gelu(x):
    return 0.5 * x * (1 + tl.tanh(tl.sqrt(2 / tl.pi) * (x + 0.044715 * tl.pow(x, 3))))

# Kernel for computing the linear layer
@triton.jit
def kernel_fma(
    a_ptr,
    b_ptr,
    c_ptr,
    bias_ptr,
    m: tl.constexpr,
    n: tl.constexpr,
    k: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    block_size_m: tl.constexpr,
    block_size_n: tl.constexpr,
    block_size_k: tl.constexpr,
    group_size_m: tl.constexpr,
    activation: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(m, block_size_m)
    num_pid_n = tl.cdiv(n, block_size_n)
    num_pid_in_group = group_size_m * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * group_size_m
    group_size_m = min(num_pid_m - first_pid_m, group_size_m)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = pid_m * block_size_m + tl.arange(0, block_size_m)
    offs_bn = pid_n * block_size_n + tl.arange(0, block_size_n)
    offs_k = tl.arange(0, block_size_k)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((block_size_m, block_size_n), dtype=tl.float32)
    for k in range(0, k, block_size_k):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        accumulator += tl.dot(a, b)
        a_ptrs += block_size_k * stride_ak
        b_ptrs += block_size_k * stride_bk

    if bias_ptr:
        bias = tl.load(bias_ptr + offs_bn[None, :])
        accumulator += bias

    if activation:
        accumulator = activation(accumulator)
    
    c = accumulator.to(tl.float16)

    offs_cm = pid_m * block_size_m + tl.arange(0, block_size_m)
    offs_cn = pid_n * block_size_n + tl.arange(0, block_size_n)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < m) & (offs_cn[None, :] < n)
    tl.store(c_ptrs, c, mask=c_mask)

# PyTorch custom Function for the linear layer
class LinearLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, bias=None, activation=None):
        assert a.shape[1] == b.shape[0], "Incompatible dimensions for matrix multiplication"
        m, k = a.shape
        k, n = b.shape
        out = torch.empty((m, n), device=a.device, dtype=a.dtype)

        grid = (triton.cdiv(m, 128) * triton.cdiv(n, 256),)
        kernel_fma[grid](
            a_ptr=a, b_ptr=b, c_ptr=out, bias_ptr=bias,
            m=m, n=n, k=k,
            stride_am=a.stride(0), stride_ak=a.stride(1),
            stride_bk=b.stride(0), stride_bn=b.stride(1),
            stride_cm=out.stride(0), stride_cn=out.stride(1),
            block_size_m=128, block_size_n=256, block_size_k=32,
            group_size_m=8, activation=activation
        )
        
        ctx.save_for_backward(a, b, bias)
        ctx.activation = activation
        return out

    @staticmethod
    def backward(ctx, grad_output):
        a, b, bias = ctx.saved_tensors
        activation = ctx.activation

        grad_a = torch.matmul(grad_output, b.t())
        grad_b = torch.matmul(a.t(), grad_output)
        grad_bias = grad_output.sum(0) if bias is not None else None

        return grad_a, grad_b, grad_bias, None

# User-facing wrapper function
def linear_layer(a, b, bias=None, activation=None):
    if activation == 'tanh':
        activation_fn = tanh
    elif activation == 'relu':
        activation_fn = relu
    elif activation == 'gelu':
        activation_fn = gelu
    elif activation == 'fast_gelu':
        activation_fn = fast_gelu
    else:
        activation_fn = None

    return LinearLayer.apply(a, b, bias, activation_fn)
