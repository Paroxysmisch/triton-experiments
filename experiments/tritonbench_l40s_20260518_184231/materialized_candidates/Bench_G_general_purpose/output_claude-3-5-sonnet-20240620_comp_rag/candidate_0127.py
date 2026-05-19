import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def quant_fused_matmul_248_kernel(
    # Pointers to matrices
    a_ptr, c_ptr, b1_ptr, b2_ptr,
    # Pointers to quantization params
    scales1_ptr, zeros1_ptr, g1_ptr,
    scales2_ptr, zeros2_ptr, g2_ptr,
    # Matrix dimensions
    M, N, K,
    # Quantization params
    bits, maxq,
    # Strides
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_scales, stride_zeros,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Fused kernel computing C = silu(A × B1) * (A × B2)
    where B1 and B2 are quantized matrices
    """
    # Number of elements per 32-bit word
    infearure_per_bits = 32 // bits
    
    # Program ID
    pid = tl.program_id(axis=0)
    
    # Calculate number of program IDs in each dimension
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    
    # Calculate group information
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Create block pointers
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Compute pointer offsets
    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    
    # Initialize accumulators
    accumulator1 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    accumulator2 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Masks for bounds checking
    a_mask = offs_am[:, None] < M
    
    # Main loop
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load group indices
        g1_idx = tl.load(g1_ptr + k * BLOCK_SIZE_K + offs_k)
        g2_idx = tl.load(g2_ptr + k * BLOCK_SIZE_K + offs_k)
        
        # Load scales and zeros
        scales1 = tl.load(scales1_ptr + g1_idx[:, None] * stride_scales)
        scales2 = tl.load(scales2_ptr + g2_idx[:, None] * stride_scales)
        
        # Load and process matrix values
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b1 = tl.load(b1_ptr + k * stride_bk * (BLOCK_SIZE_K // infearure_per_bits))
        b2 = tl.load(b2_ptr + k * stride_bk * (BLOCK_SIZE_K // infearure_per_bits))
        
        # Dequantize and compute
        shifter = (offs_k % infearure_per_bits) * bits
        b1 = ((b1 >> shifter[:, None]) & maxq) * scales1
        b2 = ((b2 >> shifter[:, None]) & maxq) * scales2
        
        # Accumulate results
        accumulator1 += tl.dot(a, b1)
        accumulator2 += tl.dot(a, b2)
        
        # Advance pointers
        a_ptrs += BLOCK_SIZE_K * stride_ak

    # Apply activation and multiplication
    c = tl.sigmoid(accumulator1) * accumulator1 * accumulator2
    
    # Store result
    c = c.to(tl.float16)
    c_ptrs = c_ptr + stride_cm * offs_am[:, None] + stride_cn * offs_bn[None, :]
    c_mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)
