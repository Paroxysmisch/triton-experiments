import triton
import triton.language as tl
import torch

@triton.jit
def batched_vecmat_kernel(
    output_ptr,   # Pointer to the output tensor [dim_m, dim_n]
    a_ptr,        # Pointer to the input matrix A [dim_m, dim_k]
    b_ptr,        # Pointer to the input tensor B [dim_m, dim_n, dim_k]
    dim_m, dim_n, dim_k,  # Dimensions of the problem
    stride_om, stride_on,  # Strides for the output tensor
    stride_am, stride_ak,  # Strides for matrix A
    stride_bm, stride_bn, stride_bk,  # Strides for tensor B
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Determine block indices for M and N dimensions
    m_index = tl.program_id(0)
    n_index = tl.program_id(1)
    
    # Compute starting offsets for the current block
    m_offset = m_index * BLOCK_M
    n_offset = n_index * BLOCK_N
    
    # Generate indices for the entire block
    m_indices = m_offset + tl.arange(0, BLOCK_M)
    n_indices = n_offset + tl.arange(0, BLOCK_N)
    k_indices = tl.arange(0, BLOCK_K)
    
    # Initialize accumulator for the block results
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Iterate over blocks in the K dimension
    num_k_blocks = tl.cdiv(dim_k, BLOCK_K)
    for k_block in range(num_k_blocks):
        k_offset = k_block * BLOCK_K
        
        # Compute pointers for current K block
        a_ptrs = a_ptr + (m_indices[:, None] * stride_am + (k_offset + k_indices[None, :]) * stride_ak)
        b_ptrs = b_ptr + (m_indices[:, None, None] * stride_bm + 
                          n_indices[None, :, None] * stride_bn + 
                          (k_offset + k_indices[None, None, :]) * stride_bk)
        
        # Load data with boundary checks using masks
        a_mask = (m_indices[:, None] < dim_m) & ((k_offset + k_indices[None, :]) < dim_k)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        
        b_mask = (m_indices[:, None, None] < dim_m) & \
                 (n_indices[None, :, None] < dim_n) & \
                 ((k_offset + k_indices[None, None, :]) < dim_k)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        
        # Broadcast A and compute element-wise product
        a_expanded = a[:, None, :]  # Shape becomes [BLOCK_M, 1, BLOCK_K]
        product = a_expanded * b    # Broadcasts to [BLOCK_M, BLOCK_N, BLOCK_K]
        
        # Accumulate the sum along the K dimension
        accumulator += tl.sum(product, axis=2)
    
    # Compute output pointers and store results
    output_ptrs = output_ptr + m_indices[:, None] * stride_om + n_indices[None, :] * stride_on
    output_mask = (m_indices[:, None] < dim_m) & (n_indices[None, :] < dim_n)
    tl.store(output_ptrs, accumulator, mask=output_mask)

def batched_vecmat(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    # Validate input dimensions
    assert A.dim() == 2, "Matrix A must be 2D"
    assert B.dim() == 3, "Tensor B must be 3D"
    dim_m, dim_k = A.size()
    b_m, dim_n, b_k = B.size()
    assert dim_m == b_m and dim_k == b_k, "Dimension mismatch"
    
    # Initialize output tensor
    output = torch.empty((dim_m, dim_n), device=A.device, dtype=A.dtype)
    
    # Configure block sizes (tune these based on your hardware)
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 128, 32
    
    # Ensure dimensions are divisible by block sizes
    assert dim_m % BLOCK_M == 0, "dim_m must be divisible by BLOCK_M"
    assert dim_n % BLOCK_N == 0, "dim_n must be divisible by BLOCK_N"
    assert dim_k % BLOCK_K == 0, "dim_k must be divisible by BLOCK_K"
    
    # Compute grid dimensions
    grid = (dim_m // BLOCK_M, dim_n // BLOCK_N)
    
    # Launch kernel with appropriate parameters
    batched_vecmat_kernel[grid](
        output, A, B,                # Data tensors
        dim_m, dim_n, dim_k,          # Dimension sizes
        output.stride(0), output.stride(1),  # Output strides
        A.stride(0), A.stride(1),     # A strides
        B.stride(0), B.stride(1), B.stride(2),  # B strides
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
    )
    
    return output
