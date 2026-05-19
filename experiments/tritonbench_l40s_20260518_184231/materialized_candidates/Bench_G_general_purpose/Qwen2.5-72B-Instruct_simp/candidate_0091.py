import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    A, B, C,  # Pointers to matrices
    M, N, K,  # Matrix dimensions
    stride_am, stride_ak,  # Strides for matrix A
    stride_bk, stride_bn,  # Strides for matrix B
    stride_cm, stride_cn,  # Strides for matrix C
    BLOCK_M: tl.constexpr,  # Block size for M dimension
    BLOCK_N: tl.constexpr,  # Block size for N dimension
    BLOCK_K: tl.constexpr,  # Block size for K dimension
    GROUP_SIZE_M: tl.constexpr  # Group size for M dimension
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) * stride_am
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) * stride_bn
    offs_k = tl.arange(0, BLOCK_K)

    A_block_ptr = tl.make_block_ptr(
        base=A,
        shape=(M, K),
        strides=(stride_am, stride_ak),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0)
    )
    B_block_ptr = tl.make_block_ptr(
        base=B,
        shape=(K, N),
        strides=(stride_bk, stride_bn),
        offsets=(0, pid_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(0, 1)
    )
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        a = tl.load(A_block_ptr)
        b = tl.load(B_block_ptr)
        acc += tl.dot(a, b)
        A_block_ptr = tl.advance(A_block_ptr, (0, BLOCK_K))
        B_block_ptr = tl.advance(B_block_ptr, (BLOCK_K, 0))

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    C_block_ptr = tl.make_block_ptr(
        base=C,
        shape=(M, N),
        strides=(stride_cm, stride_cn),
        offsets=(offs_cm, offs_cn),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0)
    )
    tl.store(C_block_ptr, acc)

### Wrapper Function to Launch the Kernel

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N', 'K']
)
@triton.jit
def matmul_dequantize_int4_s2(
    A, B, C,  # Pointers to matrices
    M, N, K,  # Matrix dimensions
    stride_am, stride_ak,  # Strides for matrix A
    stride_bk, stride_bn,  # Strides for matrix B
    stride_cm, stride_cn,  # Strides for matrix C
    BLOCK_M: tl.constexpr,  # Block size for M dimension
    BLOCK_N: tl.constexpr,  # Block size for N dimension
    BLOCK_K: tl.constexpr,  # Block size for K dimension
    GROUP_SIZE_M: tl.constexpr  # Group size for M dimension
):
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_M, BLOCK_N, BLOCK_K, GROUP_SIZE_M
    )

### Quantization and Dequantization Functions

@triton.jit
def quantize_int4(A, A_quant, scale, M, K):
    pid = tl.program_id(axis=0)
    block_size = 8
    num_blocks = tl.cdiv(K, block_size)
    block_id = pid % num_blocks
    row_id = pid // num_blocks
    offs_m = row_id * M + tl.arange(0, M)
    offs_k = block_id * block_size + tl.arange(0, block_size)
    A_block = tl.load(A + offs_m[:, None] * K + offs_k[None, :])
    A_quantized = tl.round(A_block / scale)
    A_quantized = tl.bitcast(A_quantized, tl.int4)
    A_quant_block = tl.zeros((M, block_size // 2), dtype=tl.int8)
    A_quant_block = tl.bitcast(A_quantized, A_quant_block)
    tl.store(A_quant + row_id * M * (K // 2) + block_id * (block_size // 2), A_quant_block)

@triton.jit
def unpack_int4(A_quant, A_dequant, scale, M, K):
    pid = tl.program_id(axis=0)
    block_size = 8
    num_blocks = tl.cdiv(K, block_size)
    block_id = pid % num_blocks
    row_id = pid // num_blocks
    offs_m = row_id * M + tl.arange(0, M)
    offs_k = block_id * block_size + tl.arange(0, block_size)
    A_quant_block = tl.load(A_quant + row_id * M * (K // 2) + block_id * (block_size // 2))
    A_quantized = tl.bitcast(A_quant_block, tl.int4)
    A_dequant_block = A_quantized * scale
    tl.store(A_dequant + offs_m[:, None] * K + offs_k[None, :], A_dequant_block)

### Example Usage

import torch

# Example matrices
M, N, K = 1024, 1024, 1024
A = torch.randn((M, K), device='cuda', dtype=torch.float32)
B = torch.randn((K, N), device='cuda', dtype=torch.float32)
C = torch.zeros((M, N), device='cuda', dtype=torch.float32)

# Quantize B to INT4
scale = 1.0  # Example scale factor
B_quant = torch.zeros((M, K // 2), device='cuda', dtype=torch.int8)
quantize_int4[(M * K // 8,)](A, B_quant, scale, M, K)

# Dequantize B to verify correctness
B_dequant = torch.zeros((M, K), device='cuda', dtype=torch.float32)
unpack_int4[(M * K // 8,)](B_quant, B_dequant, scale, M, K)

# Perform matrix multiplication
matmul_dequantize_int4_s2[(M * N // (128 * 128),)](A, B_dequant, C, M, N, K, A.stride(0), A.stride(1), B_dequant.stride(0), B_dequant.stride(1), C.stride(0), C.stride(1), 128, 128, 32, 8)

print(C)
