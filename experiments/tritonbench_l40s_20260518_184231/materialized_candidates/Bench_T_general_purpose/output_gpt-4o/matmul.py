import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    
    # Compute row and column indices
    row = pid // N
    col = pid % N
    
    # Create pointers for A and B
    a_ptrs = A_ptr + row * stride_am + tl.arange(0, BLOCK_SIZE) * stride_ak
    b_ptrs = B_ptr + tl.arange(0, BLOCK_SIZE)[:, None] * stride_bk + col * stride_bn
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Loop over K dimension
    for k in range(0, K, BLOCK_SIZE):
        a = tl.load(a_ptrs + k)
        b = tl.load(b_ptrs + k * stride_bk)
        acc += a @ b
    
    # Write back the result
    c_ptrs = C_ptr + row * stride_cm + col * stride_cn
    tl.store(c_ptrs, acc)

import torch

def matmul(input, other, *, out=None):
    # Determine the dimensionality of the inputs
    dim_input = input.dim()
    dim_other = other.dim()
    
    # Handle 1D dot product
    if dim_input == 1 and dim_other == 1:
        return torch.dot(input, other)
    
    # Handle 2D matrix-matrix product
    if dim_input == 2 and dim_other == 2:
        M, K = input.shape
        K, N = other.shape
        if out is None:
            out = torch.empty((M, N), device=input.device, dtype=input.dtype)
        
        grid = (M * N + BLOCK_SIZE - 1) // BLOCK_SIZE
        matmul_kernel[grid](
            input, other, out,
            M, N, K,
            input.stride(0), input.stride(1),
            other.stride(0), other.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_SIZE=32
        )
        return out
    
    # Handle 1D and 2D matrix-vector product
    if (dim_input == 1 and dim_other == 2) or (dim_input == 2 and dim_other == 1):
        if dim_input == 1:
            input, other = other, input
        return torch.mv(input, other)
    
    # Handle N-dimensional tensors (N > 2) for batched matrix multiply
    if dim_input > 2 or dim_other > 2:
        return torch.matmul(input, other)
    
    raise ValueError("Unsupported tensor dimensions for matmul operation.")
