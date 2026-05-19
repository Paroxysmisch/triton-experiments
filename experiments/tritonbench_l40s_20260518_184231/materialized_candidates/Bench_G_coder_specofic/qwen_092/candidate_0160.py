import triton
import triton.language as tl

# Constants
BLOCK_SIZE_M = 128
BLOCK_SIZE_N = 128
BLOCK_SIZE_K = 32

@triton.jit
def matmul4_kernel(
    A_ptr, B_ptr, C_ptr, A_stride_m, A_stride_k, B_stride_k, B_stride_n, C_stride_m, C_stride_n,
    M, N, K, scale_ptr, zero_point_ptr, out_dtype,
    BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K
):
    pid = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    grid_m = tl.cdiv(M, BLOCK_SIZE_M)
    grid_n = tl.cdiv(N, BLOCK_SIZE_N)
    m = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = A_ptr + m[:, None] * A_stride_m + k[None, :] * A_stride_k
    b_ptrs = B_ptr + k[:, None] * B_stride_k + n[None, :] * B_stride_n
    c_ptrs = C_ptr + m[:, None] * C_stride_m + n[None, :] * C_stride_n

    a = tl.load(a_ptrs, mask=m[:, None] < M, other=0.0)
    b = tl.load(b_ptrs, mask=k[None, :] < K, other=0.0)
    b_dequant = ((b >> 16) & 0x0F) * scale + zero_point
    c = tl.dot(a, b_dequant, allow_tf32=True)

    c = tl.math.f32_to_dtype(c, out_dtype)
    tl.store(c_ptrs, c, mask=n[:, None] < N)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=1, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=1, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32}, num_stages=1, num_warps=4),
    ],
    key=['M', 'N', 'K']
)
def matmul_dequantize_int4_gptq(
    A, B, C, A_stride_m, A_stride_k, B_stride_k, B_stride_n, C_stride_m, C_stride_n,
    M, N, K, scale, zero_point, out_dtype
):
    if C.dtype != out_dtype:
        C = C.to(out_dtype)
    if A.dtype != 'float16':
        A = A.to('float16')
    if B.dtype != 'int32':
        B = B.to('int32')
    if scale.dtype != 'float32':
        scale = scale.to('float32')
    if zero_point.dtype != 'float32':
        zero_point = zero_point.to('float32')

    grid_m = tl.cdiv(M, BLOCK_SIZE_M)
    grid_n = tl.cdiv(N, BLOCK_SIZE_N)
    grid = (grid_m, grid_n)

    matmul4_kernel[grid](A, B, C, A_stride_m, A_stride_k, B_stride_k, B_stride_n, C_stride_m, C_stride_n, M, N, K, scale, zero_point, out_dtype)

def quantize_int4(A):
    A = A.transpose()
    M, K = A.shape
    B = torch.zeros((M, K // 8), dtype=torch.int32)
    scales = torch.zeros(M, dtype=torch.float32)
    zero_points = torch.zeros(M, dtype=torch.float32)

    for i in range(M):
        a = A[i]
        min_val, max_val = a.min(), a.max()
        scale = (max_val - min_val) / 15.0
        zero_point = -min_val / scale
        scale = torch.clamp(scale, 1e-4, 1.0)
        zero_point = torch.clamp(zero_point, -127.0, 127.0)
        scales[i] = scale
        zero_points[i] = zero_point

        for j in range(K // 8):
            b = a[j * 8:j * 8 + 8]
            b_quant = ((b / scale + zero_point) + 0.5).clamp(-128, 127).to(torch.int8)
            b_quant = (b_quant & 0x0F) | ((b_quant >> 4) << 28)
            B[i, j] = b_quant

    return B, scales, zero_points
