import torch
import triton
import triton.language as tl

@triton.jit
def dequantize_kernel(
    b_ptr, b_scale_ptr, fpb_ptr,  # Pointers to input/output matrices
    K, N,                         # Matrix dimensions
    stride_bk, stride_bn,         # Strides for int8 matrix B
    stride_fpbk, stride_fpbn,     # Strides for output float matrix
    BLOCK_SIZE_N: tl.constexpr,   # Block size for N dimension
    BLOCK_SIZE_K: tl.constexpr,   # Block size for K dimension
):
    # Get program ID for the current block
    k_block_idx = tl.program_id(axis=0)  # Block index in K dimension
    n_block_idx = tl.program_id(axis=1)  # Block index in N dimension
    
    # Create offsets for the K and N dimensions
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    
    # Calculate memory offsets for input int8 matrix
    b_offs = (k_block_idx * BLOCK_SIZE_K + offs_k[:, None]) * stride_bk + \
             (n_block_idx * BLOCK_SIZE_N + offs_n[None, :]) * stride_bn
             
    # Calculate memory offsets for output float matrix
    fpb_offs = (k_block_idx * BLOCK_SIZE_K + offs_k[:, None]) * stride_fpbk + \
               (n_block_idx * BLOCK_SIZE_N + offs_n[None, :]) * stride_fpbn
               
    # Calculate offsets for scale matrix
    bs_offs = n_block_idx * BLOCK_SIZE_N + offs_n[None, :]
    
    # Create masks for boundary checking
    n_mask = n_block_idx * BLOCK_SIZE_N + offs_n[None, :] < N
    mask = (k_block_idx * BLOCK_SIZE_K + offs_k[:, None] < K) & n_mask
    
    # Load values and perform dequantization
    int_b = tl.load(b_ptr + b_offs, mask=mask, other=0.0)
    scale_b = tl.load(b_scale_ptr + bs_offs, mask=n_mask, other=0.0)
    
    # Store dequantized results
    tl.store(fpb_ptr + fpb_offs, int_b * scale_b, mask=mask)

def matmul_dequantize_int8(a, b, b_scale, out=None):
    """
    Compute matrix multiplication with int8 dequantization: C = A × (B * scale)
    
    Args:
        a: Input matrix A (float)
        b: Input matrix B (int8)
        b_scale: Scale matrix for dequantizing B
        out: Optional output matrix
    
    Returns:
        Result matrix C
    """
    # Check dimensions compatibility
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    
    # Get matrix dimensions
    M, K = a.shape
    K, N = b.shape
    
    # Initialize output matrix if not provided
    if out is None:
        c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    else:
        c = out
        
    # Allocate temporary buffer for dequantized B
    fp_b = torch.empty((K, N), device=a.device, dtype=a.dtype)
    
    # Define grid for kernel launch
    grid = lambda META: (
        triton.cdiv(K, META['BLOCK_SIZE_K']), 
        triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    
    # Launch dequantization kernel
    dequantize_kernel[grid](
        b, b_scale, fp_b,
        K, N,
        b.stride(0), b.stride(1),
        fp_b.stride(0), fp_b.stride(1),
        BLOCK_SIZE_N=32,
        BLOCK_SIZE_K=32
    )
    
    # Perform matrix multiplication
    torch.mm(a, fp_b, out=c)
    return c
