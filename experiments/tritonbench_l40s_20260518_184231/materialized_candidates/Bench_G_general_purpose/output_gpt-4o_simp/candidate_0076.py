import triton
import triton.language as tl

# Kernel for quantizing a matrix to int8 per row
@triton.jit
def quantize_int8_perrow_kernel(
    matrix_ptr, quantized_ptr, scale_ptr, 
    M, N, stride_m, stride_n,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    # Compute row index
    row_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = row_idx < M

    # Load the row from the matrix
    row = tl.load(matrix_ptr + row_idx[:, None] * stride_m + tl.arange(0, N) * stride_n, mask=mask[:, None])

    # Find the maximum value per row
    max_val = tl.max(row, axis=1)

    # Compute the scale factor
    scale = max_val / 127.0
    tl.store(scale_ptr + row_idx, scale, mask=mask)

    # Quantize the row
    quantized_row = (row / scale[:, None]).to(tl.int8)
    tl.store(quantized_ptr + row_idx[:, None] * stride_m + tl.arange(0, N) * stride_n, quantized_row, mask=mask[:, None])

def quantize_int8_perrow(matrix, M, N):
    quantized_matrix = triton.empty((M, N), dtype=tl.int8)
    scale_factors = triton.empty((M,), dtype=tl.float32)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE']),)
    quantize_int8_perrow_kernel[grid](
        matrix, quantized_matrix, scale_factors,
        M, N, matrix.stride(0), matrix.stride(1),
        BLOCK_SIZE=128
    )
    return quantized_matrix, scale_factors

# Kernel for matrix multiplication with quantized inputs
@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr, 
    scale_A_ptr, scale_B_ptr, M, N, K, 
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute start of the block
    start_m = pid_m * BLOCK_SIZE_M
    start_n = pid_n * BLOCK_SIZE_N

    # Create accumulators
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over K dimension
    for k in range(0, K, BLOCK_SIZE_K):
        # Load blocks of A and B
        a = tl.load(A_ptr + (start_m + tl.arange(0, BLOCK_SIZE_M))[:, None] * stride_am + (k + tl.arange(0, BLOCK_SIZE_K)) * stride_ak)
        b = tl.load(B_ptr + (k + tl.arange(0, BLOCK_SIZE_K))[:, None] * stride_bk + (start_n + tl.arange(0, BLOCK_SIZE_N)) * stride_bn)

        # Load scales
        scale_A = tl.load(scale_A_ptr + start_m + tl.arange(0, BLOCK_SIZE_M))
        scale_B = tl.load(scale_B_ptr + k + tl.arange(0, BLOCK_SIZE_K))

        # Convert to float32
        a = a.to(tl.float32) * scale_A[:, None]
        b = b.to(tl.float32) * scale_B[None, :]

        # Compute matrix multiplication
        acc += tl.dot(a, b)

    # Store result
    tl.store(C_ptr + (start_m + tl.arange(0, BLOCK_SIZE_M))[:, None] * stride_cm + (start_n + tl.arange(0, BLOCK_SIZE_N)) * stride_cn, acc)

def matmul_quantize_int8(A, B, M, N, K):
    # Quantize matrices A and B
    A_quantized, scale_A = quantize_int8_perrow(A, M, K)
    B_quantized, scale_B = quantize_int8_perrow(B, K, N)

    # Prepare output matrix
    C = triton.empty((M, N), dtype=tl.float32)

    # Define grid
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']), triton.cdiv(N, META['BLOCK_SIZE_N']))

    # Launch matmul kernel
    matmul_kernel[grid](
        A_quantized, B_quantized, C, 
        scale_A, scale_B, M, N, K,
        A.stride(0), A.stride(1), B.stride(0), B.stride(1), C.stride(0), C.stride(1),
        BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=32
    )
    return C

def matmul_int8(A_quantized, B_quantized, scale_A, scale_B, M, N, K):
    C = triton.empty((M, N), dtype=tl.float32)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']), triton.cdiv(N, META['BLOCK_SIZE_N']))
    matmul_kernel[grid](
        A_quantized, B_quantized, C, 
        scale_A, scale_B, M, N, K,
        A_quantized.stride(0), A_quantized.stride(1), B_quantized.stride(0), B_quantized.stride(1), C.stride(0), C.stride(1),
        BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=32
    )
    return C

def quantize_int8(matrix, axis):
    if axis == 0:
        # Quantize per row
        return quantize_int8_perrow(matrix, *matrix.shape)
    else:
        # For simplicity, we only implement per-row quantization here
        raise NotImplementedError("Only per-row quantization is implemented.")
