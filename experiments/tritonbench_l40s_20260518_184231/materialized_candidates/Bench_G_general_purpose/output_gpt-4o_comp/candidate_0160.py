import triton
import triton.language as tl
import torch

# Auxiliary function to quantize a matrix into int4 format
def quantize_int4(B, num_groups=8):
    # B is a 2D tensor with shape (K, N)
    K, N = B.shape
    assert N % num_groups == 0, "N must be divisible by num_groups"
    group_size = N // num_groups
    
    # Prepare storage for quantized values, scales, and zero points
    quantized_B = torch.zeros((K, N // 8), dtype=torch.int32)
    scales = torch.zeros((K, num_groups), dtype=torch.float32)
    zero_points = torch.zeros((K, num_groups), dtype=torch.float32)
    
    # Quantize each group
    for k in range(K):
        for g in range(num_groups):
            start = g * group_size
            end = (g + 1) * group_size
            group = B[k, start:end]
            
            min_val = group.min()
            max_val = group.max()
            scale = (max_val - min_val) / 15.0
            zero_point = min_val
            
            scales[k, g] = scale
            zero_points[k, g] = zero_point
            
            # Quantize group to int4
            quantized_group = ((group - zero_point) / scale).round().clamp(0, 15).to(torch.int32)
            
            # Pack int4 values into int32
            packed_value = 0
            for i, q in enumerate(quantized_group):
                packed_value |= (q.item() << (i * 4))
            quantized_B[k, g] = packed_value
    
    return quantized_B, scales, zero_points

# Triton kernel for matrix multiplication with int4 quantized B
@triton.autotune(configs=[
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_warps=2),
], key=['M', 'N', 'K'])
@triton.jit
def matmul4_kernel(A_ptr, B_ptr, C_ptr, scales_ptr, zero_points_ptr,
                   stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                   M, N, K, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr):
    
    pid = tl.program_id(axis=0)
    num_pid_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_pid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    A = tl.load(A_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak, mask=offs_am[:, None] < M)
    C = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        B_packed = tl.load(B_ptr + (k + offs_k)[:, None] * stride_bk + offs_bn[None, :] * stride_bn, mask=offs_bn[None, :] < N)
        scales = tl.load(scales_ptr + (k + offs_k)[:, None], mask=offs_k[:, None] < K)
        zero_points = tl.load(zero_points_ptr + (k + offs_k)[:, None], mask=offs_k[:, None] < K)
        
        B = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_N), dtype=tl.float32)
        
        for i in range(8):
            shift = i * 4
            mask = 0xF << shift
            int4_values = (B_packed & mask) >> shift
            dequantized_values = int4_values.to(tl.float32) * scales - zero_points
            B += dequantized_values

        C += tl.dot(A, B)
    
    C = C.to(tl.float16)
    tl.store(C_ptr + offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn, C, mask=offs_am[:, None] < M)

# Wrapper function to execute the kernel
def matmul_dequantize_int4_gptq(A, B_quantized, scales, zero_points, M, N, K):
    assert A.shape[0] == M and A.shape[1] == K, "A shape mismatch"
    assert B_quantized.shape[0] == K and B_quantized.shape[1] == N // 8, "B_quantized shape mismatch"
    
    # Allocate output
    C = torch.empty((M, N), dtype=torch.float16, device='cuda')

    # Define strides
    stride_am, stride_ak = A.stride()
    stride_bk, stride_bn = B_quantized.stride()
    stride_cm, stride_cn = C.stride()

    # Launch kernel
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']),)
    matmul4_kernel[grid](A, B_quantized, C, scales, zero_points,
                         stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                         M, N, K)
    return C

# Example usage
# A = torch.randn((M, K), dtype=torch.float16, device='cuda')
# B = torch.randn((K, N), dtype=torch.float32, device='cuda')
# B_quantized, scales, zero_points = quantize_int4(B)
# C = matmul_dequantize_int4_gptq(A, B_quantized, scales, zero_points, M, N, K)
