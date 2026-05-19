import torch
import triton
import triton.language as tl
import math

@triton.jit
def tanh(x):
    return tl.libdevice.tanh(x)

@triton.jit
def relu(x):
    return tl.maximum(x, 0)

@triton.jit
def gelu(x):
    # GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    sqrt_2_pi = 0.7978845608028654
    return 0.5 * x * (1 + tanh(sqrt_2_pi * (x + 0.044715 * x * x * x)))

@triton.jit
def fast_gelu(x):
    # Fast GELU approximation: x * sigmoid(1.702 * x)
    return x * tl.sigmoid(1.702 * x)

@triton.jit
def apply_activation(x, activation_type):
    if activation_type == 0:  # No activation
        return x
    elif activation_type == 1:  # ReLU
        return relu(x)
    elif activation_type == 2:  # GELU
        return gelu(x)
    elif activation_type == 3:  # Fast GELU
        return fast_gelu(x)
    elif activation_type == 4:  # Tanh
        return tanh(x)
    return x

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 4}, num_stages=5),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def kernel_fma(
    # Pointers to matrices
    A, B, C, bias,
    # Matrix dimensions
    M, N, K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. stride_am is how much to increase a_ptr
    # by to get the element one row down (A has M rows)
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_bias,
    # Meta-parameters
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    # Optional activation type
    activation_type: tl.constexpr,
):
    """
    Kernel for computing C = activation(A @ B + bias)
    A: (M, K) matrix
    B: (K, N) matrix
    C: (M, N) matrix
    bias: (N,) vector
    """
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Initialize pointers to A, B, C
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = A + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Iterate to compute matrix multiplication
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    
    # Add bias if provided
    if bias is not None:
        bias_ptrs = bias + offs_bn * stride_bias
        bias_values = tl.load(bias_ptrs)
        accumulator += bias_values[None, :]
    
    # Apply activation function
    if activation_type != 0:
        accumulator = apply_activation(accumulator, activation_type)
    
    # Store output
    c = accumulator.to(tl.float16)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    tl.store(c_ptrs, c)

class LinearLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias=None, activation='none'):
        activation_types = {
            'none': 0, 'relu': 1, 'gelu': 2,
            'fast_gelu': 3, 'tanh': 4
        }
        activation_type = activation_types.get(activation.lower(), 0)
        
        M, K = x.shape
        N = weight.shape[1]
        
        # Allocate output
        y = torch.empty((M, N), device=x.device, dtype=torch.float16)
        
        # Ensure contiguous inputs
        x = x.contiguous()
        weight = weight.contiguous()
        if bias is not None:
            bias = bias.contiguous()
        
        # Launch kernel
        grid = lambda META: (
            triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),
        )
        kernel_fma[grid](
            x, weight, y, bias,
            M, N, K,
            x.stride(0), x.stride(1),
            weight.stride(0), weight.stride(1),
            y.stride(0), y.stride(1),
            0 if bias is None else bias.stride(0),
            activation_type=activation_type,
        )
        
        return y

def linear_layer(x, weight, bias=None, activation='none'):
    """
    Applies a linear transformation to the incoming data: y = activation(x @ weight.T + bias)
    
    Args:
        x: input tensor of shape (*, in_features)
        weight: weight matrix of shape (out_features, in_features)
        bias: optional bias vector of shape (out_features,)
        activation: activation function to apply ('none', 'relu', 'gelu', 'fast_gelu', 'tanh')
    
    Returns:
        output tensor of shape (*, out_features)
    """
    return LinearLayer.apply(x, weight, bias, activation)
