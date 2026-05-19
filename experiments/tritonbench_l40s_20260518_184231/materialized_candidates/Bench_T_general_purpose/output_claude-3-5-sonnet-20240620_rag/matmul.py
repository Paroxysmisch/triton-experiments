import triton
import triton.language as tl
import torch
import math

@triton.jit
def dot_product_kernel(
    x_ptr, y_ptr, out_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    dot = tl.sum(x * y, axis=0)
    
    if pid == 0:
        tl.store(out_ptr, dot)

@triton.jit
def matmul_kernel_2d(
    a_ptr, b_ptr, c_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    pid_m = pid // num_pid_m
    pid_n = pid % num_pid_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_SIZE_K):
        mask = offs_k[None, :] < K - k
        a = tl.load(a_ptrs, mask=mask, other=0.0)
        b = tl.load(b_ptrs, mask=mask, other=0.0)
        accumulator += tl.dot(a, b)
    
    c = accumulator.to(tl.float32)
    
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    
    mask_c = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=mask_c)

def matmul(input, other, *, out=None):
    """
    Matrix multiplication implementation supporting various tensor dimensions.
    
    Args:
        input (Tensor): First input tensor
        other (Tensor): Second input tensor
        out (Tensor, optional): Output tensor
    
    Returns:
        Tensor: Result of matrix multiplication
    """
    input_dim = input.dim()
    other_dim = other.dim()
    
    # Handle 1D x 1D case (dot product)
    if input_dim == 1 and other_dim == 1:
        if input.size(0) != other.size(0):
            raise RuntimeError("Size mismatch for 1D tensors")
        if out is not None:
            raise ValueError("out parameter is not supported for 1D x 1D case")
            
        N = input.size(0)
        result = torch.empty((), dtype=input.dtype, device=input.device)
        grid = (1,)
        dot_product_kernel[grid](input, other, result, N, BLOCK_SIZE=1024)
        return result
    
    # Handle 2D x 2D case (matrix multiplication)
    elif input_dim == 2 and other_dim == 2:
        M, K = input.size()
        K2, N = other.size()
        
        if K != K2:
            raise RuntimeError(f"Size mismatch: {input.size()} x {other.size()}")
            
        if out is None:
            out = torch.empty((M, N), dtype=input.dtype, device=input.device)
        elif out.size() != (M, N):
            raise RuntimeError(f"Output tensor has incorrect size: {out.size()}, expected {(M, N)}")
        
        # Configure block sizes
        BLOCK_SIZE_M = 128
        BLOCK_SIZE_N = 128
        BLOCK_SIZE_K = 32
        
        grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),)
        
        matmul_kernel_2d[grid](
            input, other, out,
            M, N, K,
            input.stride(0), input.stride(1),
            other.stride(0), other.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
        )
        return out
    
    # Handle 1D x 2D and 2D x 1D cases (matrix-vector multiplication)
    elif (input_dim == 1 and other_dim == 2) or (input_dim == 2 and other_dim == 1):
        # Convert to 2D x 2D case and reshape result
        if input_dim == 1:
            input = input.unsqueeze(0)
            result_1d = True
        else:
            other = other.unsqueeze(1)
            result_1d = False
            
        result = matmul(input, other, out=out)
        return result.squeeze() if result_1d else result
    
    # Handle N-D cases (N > 2) with broadcasting
    else:
        raise NotImplementedError("N-dimensional (N>2) matrix multiplication not implemented in this example")
