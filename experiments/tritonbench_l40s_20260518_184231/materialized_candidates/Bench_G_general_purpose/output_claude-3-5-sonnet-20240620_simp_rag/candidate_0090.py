import torch
import triton
import triton.language as tl

@triton.jit
def dequantize_kernel(
    # Pointers to matrices
    b_ptr, b_scale_ptr, b_zp_ptr, fpb_ptr,
    # Matrix dimensions
    K, N, group_size,
    # Strides for accessing elements
    stride_bk, stride_bn,
    stride_bsk, stride_bsn,
    stride_bzpk, stride_bzpn,
    stride_fpbk, stride_fpbn,
    # Meta-parameters
    BLOCK_SIZE_K: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
):
    # Get program ID for the current block
    k_block_idx = tl.program_id(0)
    n_block_idx = tl.program_id(1)
    
    # Calculate offsets for current block
    offs_k = k_block_idx * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offs_n = n_block_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Calculate memory offsets for different matrices
    b_offs = offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    bzp_offs = offs_k[:, None] * stride_bzpk + (offs_n // group_size)[None, :] * stride_bzpn
    
    # Create masks for boundary checking
    n_mask = offs_n[None, :] < N
    k_mask = offs_k[:, None] < K
    mask = n_mask & k_mask
    
    # Load quantized values and zero points
    int32_b = tl.load(b_ptr + b_offs, mask=mask, other=0.0)
    zp_b = tl.load(b_zp_ptr + bzp_offs, mask=mask, other=0.0)
    
    # Process 8 int4 values packed in each 32-bit integer
    for i in range(8):
        # Extract int4 values and zero points
        int4_b = ((int32_b << (28 - i * 4)) >> 28 + 16) & 15
        int4_zp = ((zp_b << (28 - i * 4)) >> 28 + 16) & 15
        
        # Calculate offsets for scale and output
        bs_offs = (offs_k * 8 + i)[:, None] * stride_bsk + (offs_n // group_size)[None, :] * stride_bsn
        fpb_offs = (offs_k * 8 + i)[:, None] * stride_fpbk + offs_n[None, :] * stride_fpbn
        
        # Create mask for the current row
        k8_mask = (offs_k * 8 + i)[:, None] < K * 8
        
        # Load scale and compute dequantized values
        scale_b = tl.load(b_scale_ptr + bs_offs, mask=n_mask & k8_mask, other=0.0)
        fp_weight = (int4_b - int4_zp) * scale_b
        
        # Store dequantized values
        tl.store(fpb_ptr + fpb_offs, fp_weight, mask=n_mask & k8_mask)

def dequantize_int4(b, b_scale, b_zero_point, device, dtype, group_size):
    Kw, N = b.shape
    # Allocate output tensor
    fp_b = torch.empty((b_scale.shape[0], N), device=device, dtype=dtype)
    
    # Define grid for kernel launch
    grid = lambda META: (
        triton.cdiv(Kw, META['BLOCK_SIZE_K']),
        triton.cdiv(N, META['BLOCK_SIZE_N'])
    )
    
    # Launch kernel
    dequantize_kernel[grid](
        b, b_scale, b_zero_point, fp_b,
        Kw, N, group_size,
        b.stride(0), b.stride(1),
        b_scale.stride(0), b_scale.stride(1),
        b_zero_point.stride(0), b_zero_point.stride(1),
        fp_b.stride(0), fp_b.stride(1)
    )
    return fp_b

def matmul_dequantize_int4(a, b, b_scale, b_zero_point, group_size=128, out=None):
    # Input validation
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert b.is_contiguous(), "Matrix B must be contiguous"
    
    M, K = a.shape
    Kw, N = b.shape
    
    # Allocate output if not provided
    if out is None:
        c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    else:
        c = out
    
    # Dequantize and perform matrix multiplication
    fp_b = dequantize_int4(b, b_scale, b_zero_point, a.device, a.dtype, group_size)
    torch.mm(a, fp_b, out=c)
    return c
