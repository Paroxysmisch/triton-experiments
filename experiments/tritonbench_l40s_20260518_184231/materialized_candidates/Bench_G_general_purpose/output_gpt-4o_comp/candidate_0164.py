import triton
import triton.language as tl

@triton.jit
def tanh(x):
    return tl.tanh(x)

@triton.jit
def relu(x):
    return tl.where(x > 0, x, 0)

@triton.jit
def gelu(x):
    return 0.5 * x * (1.0 + tl.erf(x / tl.sqrt(2.0)))

@triton.jit
def fast_gelu(x):
    return 0.5 * x * (1.0 + tl.tanh(tl.sqrt(2.0 / 3.141592653589793) * (x + 0.044715 * tl.pow(x, 3))))

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64}, num_warps=2),
    ],
    key=['M', 'N', 'K']
)
@triton.jit
def kernel_fma(A_ptr, B_ptr, C_ptr, Bias_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, activation: tl.constexpr):
    # Define the block size
    BLOCK_SIZE_M = tl.cdiv(M, 128)
    BLOCK_SIZE_N = tl.cdiv(N, 128)
    BLOCK_SIZE_K = tl.cdiv(K, 32)
    
    # Define block indices
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute the start of the block
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Load A and B tiles
    A_tile = tl.load(A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    B_tile = tl.load(B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    
    # Compute matrix multiplication
    C_tile = tl.dot(A_tile, B_tile)
    
    # Add bias if provided
    if Bias_ptr is not None:
        bias_tile = tl.load(Bias_ptr + offs_n)
        C_tile += bias_tile[None, :]
    
    # Apply activation if specified
    if activation == 'tanh':
        C_tile = tanh(C_tile)
    elif activation == 'relu':
        C_tile = relu(C_tile)
    elif activation == 'gelu':
        C_tile = gelu(C_tile)
    elif activation == 'fast_gelu':
        C_tile = fast_gelu(C_tile)
    
    # Store the result
    tl.store(C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, C_tile)

import torch
from torch.autograd import Function

class LinearLayer(Function):
    @staticmethod
    def forward(ctx, A, B, bias=None, activation=None):
        M, K = A.shape
        K, N = B.shape
        C = torch.empty((M, N), device=A.device, dtype=A.dtype)
        
        # Define strides
        stride_am, stride_ak = A.stride()
        stride_bk, stride_bn = B.stride()
        stride_cm, stride_cn = C.stride()
        
        # Launch the Triton kernel
        kernel_fma[(M, N)](
            A, B, C, bias, M, N, K,
            stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
            activation=activation
        )
        
        ctx.save_for_backward(A, B, bias)
        ctx.activation = activation
        return C

    @staticmethod
    def backward(ctx, grad_output):
        A, B, bias = ctx.saved_tensors
        activation = ctx.activation
        # Compute gradients (left as an exercise)
        # ...
        return grad_A, grad_B, grad_bias, None

def linear_layer(A, B, bias=None, activation=None):
    return LinearLayer.apply(A, B, bias, activation)
