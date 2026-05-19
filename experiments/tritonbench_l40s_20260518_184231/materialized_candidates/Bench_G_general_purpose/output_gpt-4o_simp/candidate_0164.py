import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
    ],
    key=['M', 'N', 'K']
)
@triton.jit
def kernel_fma(
    A, B, C, Bias, M, N, K, stride_am, stride_ak, stride_bn, stride_bk, stride_cm, stride_cn,
    stride_bias, activation, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    pid = tl.program_id(axis=0)
    # Calculate block indices
    block_m = pid // (N // BLOCK_SIZE_N)
    block_n = pid % (N // BLOCK_SIZE_N)
    
    # Define the range of the block
    range_m = block_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    range_n = block_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    range_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Load blocks of A and B
    A_block = tl.load(A + range_m[:, None] * stride_am + range_k[None, :] * stride_ak, mask=range_m[:, None] < M)
    B_block = tl.load(B + range_k[:, None] * stride_bk + range_n[None, :] * stride_bn, mask=range_n[None, :] < N)
    
    # Initialize accumulation
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Perform matrix multiplication
    acc += tl.dot(A_block, B_block)
    
    # Add bias if provided
    if Bias is not None:
        bias = tl.load(Bias + range_n * stride_bias, mask=range_n < N)
        acc += bias[None, :]
    
    # Apply activation function
    if activation == 1:  # tanh
        acc = tl.tanh(acc)
    elif activation == 2:  # ReLU
        acc = tl.maximum(acc, 0)
    elif activation == 3:  # GELU
        acc = 0.5 * acc * (1.0 + tl.erf(acc / tl.sqrt(2.0)))
    elif activation == 4:  # Fast GELU
        acc = acc * tl.sigmoid(1.702 * acc)
    
    # Store the result
    tl.store(C + range_m[:, None] * stride_cm + range_n[None, :] * stride_cn, acc, mask=range_m[:, None] < M)

import torch
from torch.autograd import Function

class LinearLayer(Function):
    @staticmethod
    def forward(ctx, A, B, bias=None, activation=0):
        assert A.shape[1] == B.shape[0], "Incompatible dimensions for matrix multiplication"
        M, K = A.shape
        K, N = B.shape
        
        # Prepare output tensor
        C = torch.empty((M, N), device=A.device, dtype=A.dtype)
        
        # Call the Triton kernel
        grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
        kernel_fma[grid](
            A, B, C, bias, M, N, K,
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(1),
            C.stride(0), C.stride(1),
            bias.stride(0) if bias is not None else 0,
            activation,
            BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=32
        )
        
        # Save for backward if needed
        ctx.save_for_backward(A, B, bias)
        ctx.activation = activation
        
        return C

    @staticmethod
    def backward(ctx, grad_output):
        A, B, bias = ctx.saved_tensors
        activation = ctx.activation
        
        # Backpropagation logic (to be implemented as needed)
        # ...
        
        return grad_A, grad_B, grad_bias, None

def linear_layer(A, B, bias=None, activation=0):
    return LinearLayer.apply(A, B, bias, activation)
