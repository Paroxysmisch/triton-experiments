import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        {'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 32},
        {'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32},
        {'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32},
        {'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32},
        {'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32},
    ],
    key=['K', 'N'],
)
@triton.jit
def dequantize_kernel(
    b_ptr, b_scale_ptr, fpb_ptr,
    K, N,
    stride_bk, stride_bn,
    stride_fpbk, stride_fpbn,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    """Kernel for dequantizing int8 matrix B using per-column scales.
    B has shape (K, N), scales are of shape (N,), output is (K, N)
    """
    k_block_idx = tl.program_id(axis=0)
    n_block_idx = tl.program_id(axis=1)
    
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    
    # Calculate offsets for the int8 matrix B
    b_offs = (k_block_idx * BLOCK_SIZE_K + offs_k[:, None]) * stride_bk + \
             (n_block_idx * BLOCK_SIZE_N + offs_n[None, :]) * stride_bn
    # Calculate offsets for the output float matrix fp_b
    fpb_offs = (k_block_idx * BLOCK_SIZE_K + offs_k[:, None]) * stride_fpbk + \
               (n_block_idx * BLOCK_SIZE_N + offs_n[None, :]) * stride_fpbn
    # Calculate offsets for the scale vector (assumed to be of shape (N,))
    bs_offs = n_block_idx * BLOCK_SIZE_N + offs_n
    
    # Mask to handle out-of-bounds for N dimension
    n_mask = (n_block_idx * BLOCK_SIZE_N + offs_n) < N
    # Combined mask for K and N dimensions
    k_mask = (k_block_idx * BLOCK_SIZE_K + offs_k[:, None]) < K
    mask = k_mask & n_mask
    
    # Load int8 data and scale factors
    int_b = tl.load(b_ptr + b_offs, mask=mask, other=0.0)
    scale_b = tl.load(b_scale_ptr + bs_offs, mask=n_mask, other=0.0)
    
    # Dequantize and store result
    tl.store(fpb_ptr + fpb_offs, int_b * scale_b, mask=mask)

def matmul_dequantize_int8(a: torch.Tensor, b: torch.Tensor, b_scale: torch.Tensor, out: torch.Tensor = None) -> torch.Tensor:
    # Check matrix compatibility
    assert a.shape[1] == b.shape[0], f"Matrix dimensions incompatible: a.shape={a.shape}, b.shape={b.shape}"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    
    M, K = a.shape
    _, N = b.shape
    
    # Allocate output tensor if not provided
    if out is None:
        c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    else:
        c = out
    
    # Temporary buffer for dequantized B matrix
    fp_b = torch.empty((K, N), device=a.device, dtype=a.dtype)
    
    # Grid function determines launch grid based on block sizes
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
    )
    
    # Perform matrix multiplication
    torch.mm(a, fp_b, out=c)
    return c
