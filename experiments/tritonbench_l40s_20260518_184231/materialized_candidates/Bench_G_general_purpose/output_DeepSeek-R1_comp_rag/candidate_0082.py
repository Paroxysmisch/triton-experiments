import torch
import triton
import triton.language as tl

@triton.jit
def dequantize_kernel(
    # Pointers to matrices
    b_ptr, b_scale_ptr, b_zp_ptr, fpb_ptr,
    # Matrix dimensions
    K, N, group_size,
    stride_bk, stride_bn,
    stride_bsk, stride_bsn,
    stride_bzpk, stride_bzpn,
    stride_fpbk, stride_fpbn,
    # Meta-parameters
    BLOCK_SIZE_K: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
):
    """Dequantize 4-bit quantized weights into floating-point values."""
    k_block_idx = tl.program_id(0)
    n_block_idx = tl.program_id(1)
    
    offs_k = k_block_idx * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offs_n = n_block_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Calculate offsets and masks for loading quantized data and zero points
    b_offs = offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    bzp_offs = (offs_k[:, None] // (group_size // 8)) * stride_bzpk + (offs_n // group_size)[None, :] * stride_bzpn
    mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
    
    int32_b = tl.load(b_ptr + b_offs, mask=mask, other=0)
    zp_b = tl.load(b_zp_ptr + bzp_offs, mask=mask, other=0)
    
    # Process each 4-bit segment packed into int32
    for i in range(8):
        int4_b = (int32_b >> (4 * i)) & 0xF  # Extract 4-bit value
        int4_zp = (zp_b >> (4 * i)) & 0xF     # Extract 4-bit zero point
        
        # Calculate scale offsets and load scales
        scale_k = (k_block_idx * BLOCK_SIZE_K * 8 + i * BLOCK_SIZE_K) // group_size
        bs_offs = scale_k * stride_bsk + (offs_n // group_size)[None, :] * stride_bsn
        scale_b = tl.load(b_scale_ptr + bs_offs, mask=mask, other=0)
        
        # Dequantize and store
        fp_val = (int4_b - int4_zp) * scale_b
        fpb_offs = (offs_k * 8 + i)[:, None] * stride_fpbk + offs_n[None, :] * stride_fpbn
        tl.store(fpb_ptr + fpb_offs, fp_val, mask=mask)

def dequantize_int4(b: torch.Tensor, b_scale: torch.Tensor, b_zp: torch.Tensor, 
                    group_size: int = 128) -> torch.Tensor:
    """Wrapper function for dequantizing 4-bit quantized matrix."""
    assert b.is_contiguous() and b_scale.is_contiguous() and b_zp.is_contiguous(), "Inputs must be contiguous"
    K, N = b_scale.shape[0], b.shape[1]
    fp_b = torch.empty((K * 8, N), device=b.device, dtype=torch.float16)
    
    BLOCK_SIZE_K, BLOCK_SIZE_N = 32, 32  # Tunable for GPU optimization
    grid = (triton.cdiv(K, BLOCK_SIZE_K), triton.cdiv(N, BLOCK_SIZE_N))
    
    dequantize_kernel[grid](
        b, b_scale, b_zp, fp_b,
        K, N, group_size,
        b.stride(0), b.stride(1),
        b_scale.stride(0), b_scale.stride(1),
        b_zp.stride(0), b_zp.stride(1),
        fp_b.stride(0), fp_b.stride(1),
        BLOCK_SIZE_K=BLOCK_SIZE_K, BLOCK_SIZE_N=BLOCK_SIZE_N
    )
    return fp_b

def matmul_dequantize_int4(a: torch.Tensor, b: torch.Tensor, 
                           b_scale: torch.Tensor, b_zp: torch.Tensor, 
                           group_size: int = 128) -> torch.Tensor:
    """Perform matrix multiplication with dequantized 4-bit weights."""
    fp_b = dequantize_int4(b, b_scale, b_zp, group_size)
    return torch.matmul(a, fp_b)
