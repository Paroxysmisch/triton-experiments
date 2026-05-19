import triton
import triton.language as tl
import jax
import jax.numpy as jnp
import jax_triton as jt

# Define the matrix multiplication kernel
@triton.jit
def matmul_kernel(
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
    stride_bias: tl.constexpr,
    block_size_m: tl.constexpr,
    block_size_n: tl.constexpr,
    block_size_k: tl.constexpr,
    group_size_m: tl.constexpr,
    activation: tl.constexpr,
):
    """Kernel for computing the matmul C = A x B + bias with optional activation.

    A has shape (M, K), B has shape (K, N), bias has shape (N), and C has shape (M, N)
    """
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

    if bias_ptr is not None:
        bias = tl.load(bias_ptr + offs_bn)
        accumulator += bias

    if activation:
        accumulator = activation(accumulator)
    c = accumulator.to(tl.float16)

    offs_cm = pid_m * block_size_m + tl.arange(0, block_size_m)
    offs_cn = pid_n * block_size_n + tl.arange(0, block_size_n)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < m) & (offs_cn[None, :] < n)
    tl.store(c_ptrs, c, mask=c_mask)

# Define the ReLU activation function
@triton.jit
def relu(x):
    return tl.where(x >= 0, x, 0)

# Define the matrix multiplication function
def matmul(a, b, bias=None, activation=None):
    """Performs a Triton matmul with optional bias and activation."""
    block_size_m = 128
    block_size_n = 256
    block_size_k = 32
    group_size_m = 8
    m, k = a.shape
    n, _ = b.shape
    out_shape = jax.ShapeDtypeStruct(shape=(m, n), dtype=a.dtype)
    grid = (m // block_size_m * n // block_size_n,)
    return jt.triton_call(
        a,
        b,
        bias,
        kernel=matmul_kernel,
        out_shape=out_shape,
        grid=grid,
        num_warps=8,
        num_stages=3,
        m=m,
        n=n,
        k=k,
        stride_am=k,
        stride_ak=1,
        stride_bk=n,
        stride_bn=1,
        stride_cm=n,
        stride_cn=1,
        stride_bias=1,
        block_size_m=block_size_m,
        block_size_n=block_size_n,
        block_size_k=block_size_k,
        group_size_m=group_size_m,
        activation=activation
    )

# Define the LinearLayer class
class LinearLayer(jax.custom_vjp):
    @staticmethod
    def forward(ctx, x, weight, bias, activation=None):
        ctx.save_for_backward(x, weight, bias)
        ctx.activation = activation
        return matmul(x, weight, bias, activation)

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, bias = ctx.saved_tensors
        activation = ctx.activation
        grad_x = matmul(grad_output, weight.T, activation=None)
        grad_weight = matmul(x.T, grad_output, activation=None)
        grad_bias = grad_output.sum(axis=0) if bias is not None else None
        return grad_x, grad_weight, grad_bias

# Define the linear_layer function
def linear_layer(x, weight, bias=None, activation=None):
    """Performs a linear transformation with optional activation and bias."""
    return LinearLayer.apply(x, weight, bias, activation)
