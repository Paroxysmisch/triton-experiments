import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 16}, num_warps=4),
    ],
    key=['M', 'N', 'K']
)
@triton.jit
def matmul4_kernel(A, B, C, scales, zero_points, M, N, K, stride_am, stride_ak, stride_bn, stride_c, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_pid_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_pid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = A + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B + offs_k[:, None] * stride_bn + offs_bn[None, :]
    c_ptrs = C + offs_am[:, None] * stride_c + offs_bn[None, :]

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(a_ptrs, mask=offs_am[:, None] < M)
        b_int4 = tl.load(b_ptrs, mask=offs_bn[None, :] < N)

        # Dequantize B
        scale = tl.load(scales + offs_k)
        zero_point = tl.load(zero_points + offs_k)
        b = (b_int4.to(tl.float32) - zero_point) * scale

        acc += tl.dot(a, b)

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bn

    c = acc.to(tl.float16)
    tl.store(c_ptrs, c, mask=offs_am[:, None] < M)


def matmul_dequantize_int4_gptq(A, B, scales, zero_points, M, N, K):
    assert A.shape == (M, K)
    assert B.shape == (K, N)
    assert scales.shape == (K,)
    assert zero_points.shape == (K,)

    # Allocate output matrix C
    C = torch.empty((M, N), dtype=torch.float16, device='cuda')

    # Define grid dimensions
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)

    # Launch the kernel
    matmul4_kernel[grid](
        A, B, C, scales, zero_points, M, N, K,
        A.stride(0), A.stride(1), B.stride(1), C.stride(0)
    )

    return C

import torch

def quantize_int4(B, group_size=8):
    K, N = B.shape
    scales = torch.empty(K, dtype=torch.float32, device=B.device)
    zero_points = torch.empty(K, dtype=torch.int32, device=B.device)
    B_int4 = torch.empty((K, N), dtype=torch.int32, device=B.device)

    for k in range(0, K, group_size):
        b_slice = B[k:k+group_size]
        min_val, max_val = b_slice.min(), b_slice.max()
        scale = (max_val - min_val) / 15.0
        zero_point = min_val

        scales[k:k+group_size] = scale
        zero_points[k:k+group_size] = zero_point

        b_quantized = ((b_slice - zero_point) / scale).round().to(torch.int32)
        B_int4[k:k+group_size] = pack_int4(b_quantized)

    return B_int4, scales, zero_points

def pack_int4(b_quantized):
    packed = torch.zeros((b_quantized.shape[0], b_quantized.shape[1] // 8), dtype=torch.int32, device=b_quantized.device)
    for i in range(8):
        packed |= (b_quantized[:, i::8] & 0xF) << (4 * i)
    return packed
