import triton
import triton.language as tl
import torch

# Define constants for block sizes
BLOCK_SIZE_M = 128
BLOCK_SIZE_N = 128
BLOCK_SIZE_K = 32

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}),
        # Add more configurations if needed
    ],
    key=['M', 'N', 'K']
)
@triton.jit
def matmul4_kernel(
    A_ptr, B_ptr, C_ptr, scales_ptr, zeros_ptr,
    M, N, K, stride_am, stride_ak, stride_bn, stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Define the ranges for the blocks
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # Initialize the accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float16)

    # Loop over K dimension
    for k in range(0, K, BLOCK_SIZE_K):
        rk = k + tl.arange(0, BLOCK_SIZE_K)

        # Load A and B blocks
        A = tl.load(A_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak, mask=rk[None, :] < K)
        packed_B = tl.load(B_ptr + (rk // 8)[:, None] * stride_bn + rn[None, :], mask=rk[None, :] < K)

        # Unpack B
        B = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_N), dtype=tl.float16)
        for i in range(8):
            B += (((packed_B >> (i * 4)) & 0xF) - zeros_ptr[i]) * scales_ptr[i]

        # Perform the matrix multiplication
        acc += tl.dot(A, B)

    # Store the result
    tl.store(C_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn, acc)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_K': 32, 'BLOCK_SIZE_N': 128}),
        # Add more configurations if needed
    ],
    key=['K', 'N']
)
@triton.jit
def dequantize_kernel(
    B_ptr, fpb_ptr, scales_ptr, zeros_ptr,
    K, N, stride_bn, stride_fpbk, stride_fpbn,
    BLOCK_SIZE_K: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    pid_k = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Define the ranges for the blocks
    rk = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # Load packed B
    packed_B = tl.load(B_ptr + (rk // 8)[:, None] * stride_bn + rn[None, :], mask=rk[:, None] < K)

    # Unpack and dequantize
    fpb = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_N), dtype=tl.float16)
    for i in range(8):
        fpb += (((packed_B >> (i * 4)) & 0xF) - zeros_ptr[i]) * scales_ptr[i]

    # Store the result
    tl.store(fpb_ptr + rk[:, None] * stride_fpbk + rn[None, :] * stride_fpbn, fpb)

def dequantize_int4(B, scales, zeros, K, N):
    fpb = torch.empty((K, N), dtype=torch.float16, device=B.device)
    grid = (triton.cdiv(K, BLOCK_SIZE_K), triton.cdiv(N, BLOCK_SIZE_N))
    dequantize_kernel[grid](B, fpb, scales, zeros, K, N, B.stride(0), fpb.stride(0), fpb.stride(1))
    return fpb

def matmul_dequantize_int4_s1(A, B, scales, zeros, M, N, K):
    C = torch.empty((M, N), dtype=torch.float16, device=A.device)
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))
    matmul4_kernel[grid](
        A, B, C, scales, zeros,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), C.stride(0), C.stride(1)
    )
    return C

def quantize_int4(weights, num_groups):
    K, N = weights.shape
    packed_B = torch.empty((K // 8, N), dtype=torch.int32, device=weights.device)
    scales = torch.empty((num_groups,), dtype=torch.float32, device=weights.device)
    zeros = torch.empty((num_groups,), dtype=torch.int32, device=weights.device)

    # Compute group-wise statistics and quantize
    for i in range(num_groups):
        group_weights = weights[i::num_groups]
        min_val = group_weights.min()
        max_val = group_weights.max()
        scale = (max_val - min_val) / 15.0
        zero_point = -min_val / scale
        scales[i] = scale
        zeros[i] = zero_point

        # Quantize
        for k in range(0, K, 8):
            packed_value = 0
            for j in range(8):
                idx = k + j
                if idx < K:
                    quantized_value = int((group_weights[idx] - min_val) / scale)
                    packed_value |= (quantized_value & 0xF) << (j * 4)
            packed_B[k // 8, :] = packed_value

    return packed_B, scales, zeros
