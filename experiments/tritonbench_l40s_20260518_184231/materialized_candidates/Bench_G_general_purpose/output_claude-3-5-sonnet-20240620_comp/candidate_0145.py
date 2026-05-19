import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}),
        triton.Config({'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32}),
        triton.Config({'BLOCK_SIZE_N': 512, 'BLOCK_SIZE_K': 32}),
    ],
    key=['K', 'N'],
)
@triton.jit
def dequantize_kernel(
    b_ptr, b_scale_ptr, fpb_ptr,
    K, N,
    stride_bk, stride_bn,
    stride_scale_k, stride_scale_n,
    stride_fpbk, stride_fpbn,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(0)
    
    # Block dimensions
    n_idx = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    k_idx = tl.arange(0, BLOCK_SIZE_K)
    
    # Create block masks
    n_mask = n_idx < N
    k_mask = k_idx < K
    
    # Pointers to the data
    b_block_ptr = b_ptr + (n_idx[:, None] * stride_bn + k_idx[None, :] * stride_bk)
    scale_block_ptr = b_scale_ptr + (k_idx * stride_scale_k)
    fpb_block_ptr = fpb_ptr + (n_idx[:, None] * stride_fpbn + k_idx[None, :] * stride_fpbk)
    
    # Load int8 data and scale factors
    b_block = tl.load(b_block_ptr, mask=n_mask[:, None] & k_mask[None, :])
    scale = tl.load(scale_block_ptr, mask=k_mask)
    
    # Dequantize
    fpb_block = b_block * scale[None, :]
    
    # Store result
    tl.store(fpb_block_ptr, fpb_block, mask=n_mask[:, None] & k_mask[None, :])

def matmul_dequantize_int8(a: torch.Tensor, b: torch.Tensor, b_scale: torch.Tensor) -> torch.Tensor:
    """
    Performs dequantization of int8 matrix b followed by matrix multiplication with a.
    
    Args:
        a: Input matrix of shape (M, K)
        b: Quantized int8 matrix of shape (K, N)
        b_scale: Scale factors for dequantization of shape (K,)
    
    Returns:
        c: Result matrix of shape (M, N)
    """
    assert b.dtype == torch.int8, "Input b must be int8"
    assert b_scale.dtype == torch.float32, "Scale factors must be float32"
    
    M, K = a.shape
    K_, N = b.shape
    assert K == K_, f"Incompatible dimensions: {K} != {K_}"
    
    # Allocate output for dequantized matrix
    fpb = torch.empty((K, N), device=b.device, dtype=torch.float32)
    
    # Launch dequantization kernel
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE_N']),)
    
    dequantize_kernel[grid](
        b_ptr=b, 
        b_scale_ptr=b_scale,
        fpb_ptr=fpb,
        K=K, N=N,
        stride_bk=b.stride(0),
        stride_bn=b.stride(1),
        stride_scale_k=b_scale.stride(0),
        stride_scale_n=0,  # Not used
        stride_fpbk=fpb.stride(0),
        stride_fpbn=fpb.stride(1),
    )
    
    # Perform matrix multiplication
    c = torch.mm(a, fpb)
    return c
