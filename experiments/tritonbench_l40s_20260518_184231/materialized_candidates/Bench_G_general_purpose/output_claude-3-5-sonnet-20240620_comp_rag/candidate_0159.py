import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul4_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # Strides
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Scale and zero point for dequantization
    scale_ptr, zero_ptr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Matrix multiplication block
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    
    # Calculate pid for m and n
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    # Calculate offsets
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Initialize pointers
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    
    # Load scales and zero points
    scales = tl.load(scale_ptr + pid_n)
    zeros = tl.load(zero_ptr + pid_n)
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Main loop
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load and dequantize matrix A
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        
        # Load and dequantize matrix B (int4)
        b_packed = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0)
        
        # Unpack int4 values (two values per byte)
        b_low = b_packed & 0xF
        b_high = (b_packed >> 4) & 0xF
        
        # Dequantize
        b_low = (b_low.to(tl.float32) - zeros) * scales
        b_high = (b_high.to(tl.float32) - zeros) * scales
        
        # Compute matrix multiplication
        accumulator += tl.dot(a, b_low)
        accumulator += tl.dot(a, b_high)
        
        # Advance pointers
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Store output
    c = accumulator.to(tl.float16)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

def quantize_int4(matrix, group_size=128):
    """Quantize matrix to int4 format with GPTQ"""
    device = matrix.device
    orig_shape = matrix.shape
    matrix = matrix.reshape(-1, group_size)
    
    # Calculate scales and zero points
    min_vals = matrix.min(dim=1, keepdim=True)[0]
    max_vals = matrix.max(dim=1, keepdim=True)[0]
    scales = (max_vals - min_vals) / 15
    zeros = min_vals
    
    # Quantize
    matrix_q = ((matrix - zeros) / scales * 15).round().clamp(0, 15).to(torch.int8)
    
    # Pack two int4 values into one int8
    matrix_packed = matrix_q[:, ::2] | (matrix_q[:, 1::2] << 4)
    
    return matrix_packed.reshape(orig_shape[0], -1), scales.squeeze(), zeros.squeeze()

def matmul_dequantize_int4_gptq(a, b_quantized, scales, zeros):
    """Wrapper function for int4 matrix multiplication"""
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert b_quantized.is_contiguous(), "Matrix B must be contiguous"
    
    M, K = a.shape
    K_packed = b_quantized.shape[0]
    N = b_quantized.shape[1] * 2  # Each int8 contains two int4 values
    
    # Create output matrix
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    
    # Launch kernel
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
    
    matmul4_kernel[grid](
        a, b_quantized, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b_quantized.stride(0), b_quantized.stride(1),
        c.stride(0), c.stride(1),
        scales, zeros
    )
    
    return c
