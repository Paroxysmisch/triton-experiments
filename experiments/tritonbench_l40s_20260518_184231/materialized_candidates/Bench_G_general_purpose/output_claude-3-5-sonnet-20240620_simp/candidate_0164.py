import torch
import triton
import triton.language as tl
import math

@triton.jit
def tanh(x):
    return tl.tanh(x)

@triton.jit
def relu(x):
    return tl.max(x, 0.0)

@triton.jit
def gelu(x):
    # GELU(x) = x * Φ(x)
    # where Φ(x) is the cumulative distribution function of the standard normal distribution
    return x * 0.5 * (1.0 + tl.tanh(0.797885 * x * (1.0 + 0.044715 * x * x)))

@triton.jit
def fast_gelu(x):
    # Fast approximation of GELU
    return x * tl.sigmoid(1.702 * x)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4}, num_stages=5),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def kernel_fma(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr, bias_ptr,
    # Matrix dimensions
    M, N, K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. stride_am is how much to increase a_ptr
    # by to get the element one row down (A has M rows)
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    # Activation function type
    activation_type: tl.constexpr,
):
    """
    Compute: C = activation(A @ B + bias)
    """
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Initialize pointers to A, B, C
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Iterate to compute matrix multiplication
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
        
    # Add bias if provided
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + offs_bn)
        accumulator += bias

    # Apply activation function
    if activation_type == 1:  # ReLU
        accumulator = relu(accumulator)
    elif activation_type == 2:  # GELU
        accumulator = gelu(accumulator)
    elif activation_type == 3:  # Fast GELU
        accumulator = fast_gelu(accumulator)
    elif activation_type == 4:  # tanh
        accumulator = tanh(accumulator)

    # Store output
    c_ptrs = c_ptr + stride_cm * offs_am[:, None] + stride_cn * offs_bn[None, :]
    tl.store(c_ptrs, accumulator)

class LinearLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias=None, activation='none'):
        # Dictionary to map activation functions to integers
        activation_map = {
            'none': 0,
            'relu': 1,
            'gelu': 2,
            'fast_gelu': 3,
            'tanh': 4
        }
        
        M, K = x.shape
        N = weight.shape[1]
        
        # Allocate output
        y = torch.empty((M, N), device=x.device, dtype=x.dtype)
        
        # Ensure contiguous inputs
        x = x.contiguous()
        weight = weight.contiguous()
        if bias is not None:
            bias = bias.contiguous()
        
        # Launch kernel
        grid = lambda META: (
            triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
        )
        
        kernel_fma[grid](
            x, weight, y, bias,
            M, N, K,
            x.stride(0), x.stride(1),
            weight.stride(0), weight.stride(1),
            y.stride(0), y.stride(1),
            activation_type=activation_map[activation],
        )
        
        # Save for backward
        ctx.save_for_backward(x, weight, bias)
        ctx.activation = activation
        
        return y

    @staticmethod
    def backward(ctx, grad_output):
        # Implement backward pass if needed
        # This is left as an exercise
        raise NotImplementedError("Backward pass not implemented yet")

def linear_layer(x, weight, bias=None, activation='none'):
    """
    Wrapper function for the custom linear layer.
    
    Args:
        x: Input tensor of shape (M, K)
        weight: Weight matrix of shape (K, N)
        bias: Optional bias vector of shape (N,)
        activation: Activation function ('none', 'relu', 'gelu', 'fast_gelu', 'tanh')
    
    Returns:
        Output tensor of shape (M, N)
    """
    return LinearLayer.apply(x, weight, bias, activation)
