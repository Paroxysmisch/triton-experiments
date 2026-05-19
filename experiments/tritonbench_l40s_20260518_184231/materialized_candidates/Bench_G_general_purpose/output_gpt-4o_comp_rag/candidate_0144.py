import torch
import triton
import triton.language as tl

@triton.jit
def dequantize_kernel(
    b_ptr, b_scale_ptr, fpb_ptr,
    K, N,
    stride_bk, stride_bn,
    stride_fpbk, stride_fpbn,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    """Kernel for dequantizing int8 matrix B using scale factors and storing the result in fpb."""
    k_block_idx = tl.program_id(axis=0)
    n_block_idx = tl.program_id(axis=1)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    
    # Calculate offsets for B and fpB
    b_offs = (k_block_idx * BLOCK_SIZE_K + offs_k[:, None]) * stride_bk + \
             (n_block_idx * BLOCK_SIZE_N + offs_n[None, :]) * stride_bn
    fpb_offs = (k_block_idx * BLOCK_SIZE_K + offs_k[:, None]) * stride_fpbk + \
               (n_block_idx * BLOCK_SIZE_N + offs_n[None, :]) * stride_fpbn
    
    # Calculate offsets for scale
    bs_offs = n_block_idx * BLOCK_SIZE_N + offs_n[None, :]
    
    # Create masks to handle out-of-bounds
    n_mask = n_block_idx * BLOCK_SIZE_N + offs_n[None, :] < N
    mask = (k_block_idx * BLOCK_SIZE_K + offs_k[:, None] < K) & n_mask
    
    # Load int8 values from B and scale factors
    int_b = tl.load(b_ptr + b_offs, mask=mask, other=0.0)
    scale_b = tl.load(b_scale_ptr + bs_offs, mask=n_mask, other=0.0)
    
    # Dequantize and store the result in fpB
    tl.store(fpb_ptr + fpb_offs, int_b * scale_b, mask=mask)

def matmul_dequantize_int8(a, b, b_scale, out=None):
    """Performs matrix multiplication with dequantization of int8 matrix B."""
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    
    M, K = a.shape
    K, N = b.shape
    
    if out is None:
        c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    else:
        c = out
    
    fp_b = torch.empty((K, N), device=a.device, dtype=torch.float32)
    
    # Define the grid for kernel launch
    grid = lambda META: (
        triton.cdiv(K, META['BLOCK_SIZE_K']), triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    
    # Launch the dequantize kernel
    dequantize_kernel[grid](
        b, b_scale, fp_b,
        K, N,
        b.stride(0), b.stride(1),
        fp_b.stride(0), fp_b.stride(1),
        BLOCK_SIZE_N=128, BLOCK_SIZE_K=32  # Example block sizes
    )
    
    # Perform matrix multiplication
    torch.mm(a, fp_b, out=c)
    
    return c
