import triton
import triton.language as tl

# Triton kernel to quantize a floating-point matrix to int8 per row
@triton.jit
def quantize_int8_perrow_kernel(
    X_ptr,  # Pointer to the input matrix
    X_quant_ptr,  # Pointer to the output quantized matrix
    scale_ptr,  # Pointer to the scaling factors
    stride_xr,  # Stride of the input matrix along the row
    stride_xc,  # Stride of the input matrix along the column
    stride_qr,  # Stride of the quantized matrix along the row
    stride_qc,  # Stride of the quantized matrix along the column
    stride_s,  # Stride of the scaling factors
    M,  # Number of rows in the input matrix
    N,  # Number of columns in the input matrix
    BLOCK_SIZE: tl.constexpr  # Block size for parallelization
):
    pid = tl.program_id(axis=0)
    row = pid

    if row < M:
        X_row = X_ptr + row * stride_xr
        X_quant_row = X_quant_ptr + row * stride_qr
        scale = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        max_val = tl.zeros((1,), dtype=tl.float32)

        for col in range(0, N, BLOCK_SIZE):
            cols = col + tl.arange(0, BLOCK_SIZE)
            mask = cols < N
            x = tl.load(X_row + cols * stride_xc, mask=mask, other=0.0)
            max_val = tl.max(max_val, tl.max(x, axis=0))

        scale = 127.0 / max_val
        tl.store(scale_ptr + row * stride_s, scale)

        for col in range(0, N, BLOCK_SIZE):
            cols = col + tl.arange(0, BLOCK_SIZE)
            mask = cols < N
            x = tl.load(X_row + cols * stride_xc, mask=mask, other=0.0)
            x_quant = tl.cast(tl.round(x * scale), tl.int8)
            tl.store(X_quant_row + cols * stride_qc, x_quant, mask=mask)

# Helper function to perform quantization per row
def quantize_int8_perrow(X, M, N):
    X_quant = tl.zeros((M, N), dtype=tl.int8)
    scale = tl.zeros((M,), dtype=tl.float32)
    grid = (M,)
    quantize_int8_perrow_kernel[grid](
        X, X_quant, scale, X.stride(0), X.stride(1), X_quant.stride(0), X_quant.stride(1), scale.stride(0), M, N, 128
    )
    return X_quant, scale

# Triton kernel for matrix multiplication with quantized matrices
@triton.jit
def matmul_kernel(
    A_ptr,  # Pointer to the input matrix A
    B_ptr,  # Pointer to the input matrix B
    C_ptr,  # Pointer to the output matrix C
    A_scale_ptr,  # Pointer to the scaling factors of A
    B_scale_ptr,  # Pointer to the scaling factors of B
    stride_ar,  # Stride of A along the row
    stride_ac,  # Stride of A along the column
    stride_br,  # Stride of B along the row
    stride_bc,  # Stride of B along the column
    stride_cr,  # Stride of C along the row
    stride_cc,  # Stride of C along the column
    M,  # Number of rows in A
    N,  # Number of columns in A (and rows in B)
    K,  # Number of columns in B
    BLOCK_SIZE_M: tl.constexpr,  # Block size for rows of A
    BLOCK_SIZE_N: tl.constexpr,  # Block size for columns of B
    BLOCK_SIZE_K: tl.constexpr  # Block size for columns of A (and rows of B)
):
    pid = tl.program_id(axis=0)
    pid_m = pid // N
    pid_n = pid % N
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    A = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.int8)
    B = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_N), dtype=tl.int8)
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        A = tl.load(A_ptr + (offs_am[:, None] * stride_ar + (k + offs_k[None, :]) * stride_ac))
        B = tl.load(B_ptr + ((k + offs_k[:, None]) * stride_br + offs_bn[None, :]) * stride_bc)
        A = tl.cast(A, tl.float32) * tl.load(A_scale_ptr + offs_am[:, None])
        B = tl.cast(B, tl.float32) * tl.load(B_scale_ptr + offs_k[None, :])
        acc += tl.dot(A, B)

    C = tl.store(C_ptr + (offs_am[:, None] * stride_cr + offs_bn[None, :]) * stride_cc, acc)

# Wrapper function to perform matrix multiplication with quantized matrices
def matmul_quantize_int8(A, B, M, N, K):
    A_quant, A_scale = quantize_int8_perrow(A, M, N)
    B_quant, B_scale = quantize_int8_perrow(B, N, K)
    C = tl.zeros((M, K), dtype=tl.float32)
    grid = (M * K // 128,)
    matmul_kernel[grid](
        A_quant, B_quant, C, A_scale, B_scale, A_quant.stride(0), A_quant.stride(1), B_quant.stride(0), B_quant.stride(1), C.stride(0), C.stride(1), M, N, K, 128, 128, 128
    )
    return C

# Triton kernel for matrix multiplication with already quantized matrices
@triton.jit
def matmul_int8_kernel(
    A_ptr,  # Pointer to the input matrix A
    B_ptr,  # Pointer to the input matrix B
    C_ptr,  # Pointer to the output matrix C
    A_scale_ptr,  # Pointer to the scaling factors of A
    B_scale_ptr,  # Pointer to the scaling factors of B
    stride_ar,  # Stride of A along the row
    stride_ac,  # Stride of A along the column
    stride_br,  # Stride of B along the row
    stride_bc,  # Stride of B along the column
    stride_cr,  # Stride of C along the row
    stride_cc,  # Stride of C along the column
    M,  # Number of rows in A
    N,  # Number of columns in A (and rows in B)
    K,  # Number of columns in B
    BLOCK_SIZE_M: tl.constexpr,  # Block size for rows of A
    BLOCK_SIZE_N: tl.constexpr,  # Block size for columns of B
    BLOCK_SIZE_K: tl.constexpr  # Block size for columns of A (and rows of B)
):
    pid = tl.program_id(axis=0)
    pid_m = pid // N
    pid_n = pid % N
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    A = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.int8)
    B = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_N), dtype=tl.int8)
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        A = tl.load(A_ptr + (offs_am[:, None] * stride_ar + (k + offs_k[None, :]) * stride_ac))
        B = tl.load(B_ptr + ((k + offs_k[:, None]) * stride_br + offs_bn[None, :]) * stride_bc)
        A = tl.cast(A, tl.float32) * tl.load(A_scale_ptr + offs_am[:, None])
        B = tl.cast(B, tl.float32) * tl.load(B_scale_ptr + offs_k[None, :])
        acc += tl.dot(A, B)

    C = tl.store(C_ptr + (offs_am[:, None] * stride_cr + offs_bn[None, :]) * stride_cc, acc)

# Wrapper function to perform matrix multiplication with already quantized matrices
def matmul_int8(A, B, A_scale, B_scale, M, N
