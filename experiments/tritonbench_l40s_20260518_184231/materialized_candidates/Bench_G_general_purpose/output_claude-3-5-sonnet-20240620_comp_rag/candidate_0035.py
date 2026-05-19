import triton
import triton.language as tl
import torch

@triton.jit
def batched_vecmat_kernel(
    # Pointers to matrices
    A, B, Output,
    # Matrix dimensions
    dim_m, dim_n, dim_k,
    # Matrix strides
    stride_am, stride_ak,
    stride_bm, stride_bn, stride_bk,
    stride_om, stride_on,
    # Block sizes (constants)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr
):
    # Program ID
    m_index = tl.program_id(0)
    n_index = tl.program_id(1)
    
    # Calculate offsets for this block
    offs_m = m_index * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = n_index * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Initialize accumulator
    vecmat = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Pointers for current block
    a_ptrs = A + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_m[:, None] * stride_bm + offs_n[None, :] * stride_bn + offs_k[:] * stride_bk)
    
    # Iterate over k dimension
    for k in range(0, tl.cdiv(dim_k, BLOCK_K)):
        k_mask = offs_k[None, :] < (dim_k - k * BLOCK_K)
        
        # Load and broadcast vector A
        a = tl.load(a_ptrs, mask=k_mask, other=0.0)
        a = tl.broadcast_to(a, (BLOCK_M, BLOCK_N))
        
        # Load matrix slice B
        b = tl.load(b_ptrs, mask=k_mask, other=0.0)
        
        # Accumulate product
        vecmat += a * b
        
        # Move pointers to next k block
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    
    # Write output with mask
    output_mask = (offs_m[:, None] < dim_m) & (offs_n[None, :] < dim_n)
    output_ptrs = Output + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(output_ptrs, vecmat, mask=output_mask)

def batched_vecmat(A: torch.Tensor, B: torch.Tensor):
    """
    Compute batched vector-matrix multiplication.
    Args:
        A: Input vector of shape (dim_m, dim_k)
        B: Input matrix of shape (dim_m, dim_n, dim_k)
    Returns:
        Output tensor of shape (dim_m, dim_n)
    """
    assert A.is_cuda and B.is_cuda
    assert A.dim() == 2 and B.dim() == 3
    
    dim_m, dim_k = A.shape
    b_dim_m, dim_n, b_dim_k = B.shape
    
    assert dim_m == b_dim_m and dim_k == b_dim_k, "Incompatible dimensions"
    
    # Block sizes
    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_K = 32
    
    # Ensure dimensions are divisible by block sizes
    assert dim_m % BLOCK_M == 0
    assert dim_n % BLOCK_N == 0
    assert dim_k % BLOCK_K == 0
    
    # Allocate output
    output = torch.empty((dim_m, dim_n), device=A.device, dtype=A.dtype)
    
    # Calculate grid dimensions
    grid = (
        triton.cdiv(dim_m, BLOCK_M),
        triton.cdiv(dim_n, BLOCK_N),
    )
    
    # Launch kernel
    batched_vecmat_kernel[grid](
        A, B, output,
        dim_m, dim_n, dim_k,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1), B.stride(2),
        output.stride(0), output.stride(1),
        BLOCK_M, BLOCK_N, BLOCK_K
    )
    
    return output
