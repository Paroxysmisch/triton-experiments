import triton
import triton.language as tl
import torch

@triton.jit
def matmul_kernel(
    A, B, C, scales, zero_points, M, N, K, 
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, 
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    SPLIT_K: tl.constexpr
):
    pid = tl.program_id(0)
    pid_m = pid // (N // BLOCK_SIZE_N)
    pid_n = pid % (N // BLOCK_SIZE_N)
    
    # Offsets
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension
    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(A + (offs_am[:, None] * stride_am + (offs_k + k)[None, :] * stride_ak), mask=offs_am[:, None] < M)
        b = tl.load(B + ((offs_k + k)[:, None] * stride_bk + offs_bn[None, :] * stride_bn), mask=offs_bn[None, :] < N)
        
        # Dequantize B
        scale = tl.load(scales + (offs_k + k))
        zero_point = tl.load(zero_points + (offs_k + k))
        b = (b - zero_point) * scale
        
        # Compute
        acc += tl.dot(a, b)
    
    # Store result
    c = C + (offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn)
    if SPLIT_K > 1:
        tl.atomic_add(c, acc, mask=offs_am[:, None] < M)
    else:
        tl.store(c, acc, mask=offs_am[:, None] < M)

# Quantization and dequantization functions
def quantize_int4(weights):
    # Quantize weights to INT4
    min_val, max_val = weights.min(), weights.max()
    scale = (max_val - min_val) / 15.0
    zero_point = -min_val / scale
    int_weights = ((weights / scale) + zero_point).round().clamp(0, 15).to(torch.int32)
    
    # Pack INT4 values into INT32
    packed_weights = torch.zeros((int_weights.numel() + 7) // 8, dtype=torch.int32)
    for i in range(int_weights.numel()):
        packed_weights[i // 8] |= (int_weights[i] & 0xF) << (4 * (i % 8))
    
    return packed_weights, scale, zero_point

def unpack_int4(packed_weights, scale, zero_point, original_shape):
    # Unpack INT4 values
    int_weights = torch.zeros(original_shape.numel(), dtype=torch.int32)
    for i in range(int_weights.numel()):
        int_weights[i] = (packed_weights[i // 8] >> (4 * (i % 8))) & 0xF
    
    # Dequantize
    return (int_weights - zero_point) * scale

def matmul_dequantize_int4_s2(A, B_packed, scales, zero_points, M, N, K, BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=32, SPLIT_K=1):
    # Prepare output tensor
    C = torch.empty((M, N), dtype=torch.float32, device='cuda')
    
    # Launch kernel
    grid = (M // BLOCK_SIZE_M, N // BLOCK_SIZE_N)
    matmul_kernel[grid](
        A, B_packed, C, scales, zero_points, M, N, K,
        A.stride(0), A.stride(1), B_packed.stride(0), B_packed.stride(1), C.stride(0), C.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
        SPLIT_K=SPLIT_K
    )
    
    return C
