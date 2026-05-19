import triton
import triton.language as tl

# Define activation functions
@triton.jit
def tanh(x):
    return tl.tanh(x)

@triton.jit
def relu(x):
    return tl.relu(x)

@triton.jit
def gelu(x):
    return 0.5 * x * (1 + tl.tanh(0.7978845608 * (x + 0.044715 * x * x * x)))

@triton.jit
def fast_gelu(x):
    return 0.5 * x * (1 + tl.sign(x) * tl.sqrt(1 + 0.2888 * x * x))

# Define the kernel function
@triton.jit
def kernel_fma(A, B, C, bias, activation, M, N, K, stride_a_row, stride_a_col, stride_b_row, stride_b_col, stride_c_row, stride_c_col, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = num_pid_m * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * num_pid_m
    group_size = num_pid_in_group
    pid_m = first_pid_m + (pid % num_pid_m)
    pid_n = (pid % group_size) // num_pid_m

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = A + (offs_am[:, None] * stride_a_row + offs_k[None, :] * stride_a_col)
    b_ptrs = B + (offs_k[:, None] * stride_b_row + offs_bn[None, :] * stride_b_col)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_a_col
        b_ptrs += BLOCK_K * stride_b_row

    c_ptrs = C + (offs_am[:, None] * stride_c_row + offs_bn[None, :] * stride_c_col)
    if bias is not None:
        bias_ptrs = bias + offs_bn
        bias_values = tl.load(bias_ptrs)
        accumulator += bias_values

    if activation == 0:
        pass
    elif activation == 1:
        accumulator = tanh(accumulator)
    elif activation == 2:
        accumulator = relu(accumulator)
    elif activation == 3:
        accumulator = gelu(accumulator)
    elif activation == 4:
        accumulator = fast_gelu(accumulator)

    tl.store(c_ptrs, accumulator)

import torch
import triton
import triton.language as tl

class LinearLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B, bias=None, activation=None):
        M, K = A.shape
        K, N = B.shape
        assert K == B.shape[0], "Incompatible dimensions"
        
        # Determine the activation function
        activation_map = {
            'tanh': 1,
            'relu': 2,
            'gelu': 3,
            'fast_gelu': 4
        }
        activation_code = activation_map.get(activation, 0)
        
        # Allocate output tensor
        C = torch.empty((M, N), device=A.device, dtype=A.dtype)
        
        # Launch the kernel
        grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
        kernel_fma[grid](A, B, C, bias, activation_code, M, N, K, A.stride(0), A.stride(1), B.stride(0), B.stride(1), C.stride(0), C.stride(1), BLOCK_M=16, BLOCK_N=16, BLOCK_K=16)
        
        # Save inputs for backward pass
        ctx.save_for_backward(A, B, bias)
        ctx.activation = activation_code
        
        return C

    @staticmethod
    def backward(ctx, grad_output):
        A, B, bias = ctx.saved_tensors
        activation_code = ctx.activation
        M, K = A.shape
        K, N = B.shape
        
        # Allocate gradients
        grad_A = torch.zeros_like(A)
        grad_B = torch.zeros_like(B)
        if bias is not None:
            grad_bias = torch.zeros_like(bias)
        
        # Launch the kernel for gradient computation
        grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
        kernel_fma[grid](grad_output, B.t(), grad_A, None, 0, M, K, N, grad_output.stride(0), grad_output.stride(1), B.stride(1), B.stride(0), grad_A.stride(0), grad_A.stride(1), BLOCK_M=16, BLOCK_N=16, BLOCK_K=16)
        kernel_fma[grid](A, grad_output, grad_B, None, 0, K, N, M, A.stride(0), A.stride(1), grad_output.stride(0), grad_output.stride(1), grad_B.stride(0), grad_B.stride(1), BLOCK_M=16, BLOCK_N=16, BLOCK_K=16)
        
        if bias is not None:
            grad_bias = grad_output.sum(dim=0)
        
        return grad_A, grad_B, grad_bias, None

def linear_layer(A, B, bias=None, activation=None, save_pre_activation=False):
    if save_pre_activation:
        pre_activation = A @ B
        if bias is not None:
            pre_activation += bias
        if activation is not None:
            activation_map = {
                'tanh': torch.tanh,
                'relu': torch.relu,
                'gelu': torch.nn.functional.gelu,
                'fast_gelu': lambda x: 0.5 * x * (1 + torch.tanh(0.7978845608 * (x + 0.044715 * x * x * x)))
            }
            pre_activation = activation_map[activation](pre_activation)
        return pre_activation
    else:
        return LinearLayer.apply(A, B, bias, activation)

import torch

# Create input tensors
A = torch.randn(128, 256, device='cuda')
B = torch.randn(256, 64, device='cuda')
bias = torch.randn(64, device='cuda')

# Apply the linear layer with ReLU activation
output = linear_layer(A, B, bias, activation='relu')

# Print the output
print(output)
