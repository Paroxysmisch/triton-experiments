import torch
import triton
import triton.language as tl

# Triton kernel for 4-bit quantized matrix multiplication
@triton.jit
def matmul4_kernel(
    x,
    qweight,
    scales,
    qzeros,
    qweight_zero_pt,
    M,
    N,
    K,
    C,
    stride_x_m,
    stride_x_k,
    stride_qweight_m,
    stride_qweight_n,
    stride_qweight_k,
    stride_scales_n,
    stride_qzeros_n,
    stride_qweight_zero_pt_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Compute the matrix multiplication C = x * weight.
    x is of shape (M, K), weight is of shape (K//8, N) and C is of shape (M, N)
    The input matrices are quantized and dequantized during the multiplication
    """
    # block indices
    m = tl.program_id(0)
    n = tl.program_id(1)

    # offset to the current block
    offset_x_m = m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offset_x_k = tl.arange(0, BLOCK_SIZE_K)
    offset_weight_n = n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offset_weight_k = tl.arange(0, BLOCK_SIZE_K)

    # pointers to the current block of input and output
    x_ptrs = x + offset_x_m[:, None] * stride_x_m + offset_x_k[None, :] * stride_x_k
    weight_ptrs = (
        qweight
        + offset_weight_k[:, None] * stride_qweight_k
        + offset_weight_n[None, :] * stride_qweight_n
    )
    scales_ptrs = scales + offset_weight_n * stride_scales_n
    qzeros_ptrs = qzeros + offset_weight_n * stride_qzeros_n
    qweight_zero_pt_ptrs = (
        qweight_zero_pt + offset_weight_n * stride_qweight_zero_pt_n
    )

    # initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # loop over k
    for k in range(0, K, BLOCK_SIZE_K):
        # load x block and weight block
        x_block = tl.load(
            x_ptrs,
            mask=(offset_x_m[:, None] < M) & (offset_x_k[None, :] < K),
            other=0.0,
        )
        weight = tl.load(
            weight_ptrs,
            mask=(offset_weight_k[:, None] < K) & (offset_weight_n[None, :] < N),
            other=0.0,
        )
        scale = tl.load(
            scales_ptrs,
            mask=offset_weight_n < N,
        )
        qzero = tl.load(
            qzeros_ptrs,
            mask=offset_weight_n < N,
        )
        qweight_zero_pt = tl.load(
            qweight_zero_pt_ptrs,
            mask=offset_weight_n < N,
        )
        # dequantize weight
        weight = (weight - qweight_zero_pt) * scale + qzero

        acc += tl.dot(x_block, weight)

        # advance the pointers
        x_ptrs += BLOCK_SIZE_K * stride_x_k
        weight_ptrs += BLOCK_SIZE_K * stride_qweight_k

    # store the result
    offset_c_m = m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offset_c_n = n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C + offset_c_m[:, None] * stride_x_m + offset_c_n[None, :] * stride_x_k

    mask = (offset_c_m[:, None] < M) & (offset_c_n[None, :] < N)

    tl.store(c_ptrs, acc.to(C.dtype.element_ty), mask=mask)


# Function to invoke the Triton kernel for 4-bit quantized matrix multiplication
def matmul_dequantize_int4_gptq(x, qweight, scales, qzeros, qweight_zero_pt):
    M, K = x.shape
    K, N = qweight.shape
    C = torch.empty((M, N), device=x.device, dtype=x.dtype)
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    matmul4_kernel[grid](
        x,
        qweight,
        scales,
        qzeros,
        qweight_zero_pt,
        M,
        N,
        K,
        C,
        x.stride(0),
        x.stride(1),
        qweight.stride(0),
        qweight.stride(1),
        qweight.stride(2),
        scales.stride(0),
        qzeros.stride(0),
        qweight_zero_pt.stride(0),
    )
    return C


# Triton kernel for 4-bit quantized matrix multiplication with kernel splitting
@triton.jit
def matmul_kernel(
    x,
    qweight,
    scales,
    qzeros,
    qweight_zero_pt,
    M,
    N,
    K,
    C,
    stride_x_m,
    stride_x_k,
    stride_qweight_m,
    stride_qweight_n,
    stride_qweight_k,
    stride_scales_n,
    stride_qzeros_n,
    stride_qweight_zero_pt_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    """
    Compute the matrix multiplication C = x * weight.
    x is of shape (M, K), weight is of shape (K//8, N) and C is of shape (M, N)
    The input matrices are quantized and dequantized during the multiplication
    """
    # block indices
    m = tl.program_id(0)
    n = tl.program_id(1)

    # offset to the current block
    offset_x_m = m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offset_x_k = tl.arange(0, BLOCK_SIZE_K)
    offset_weight_n = n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offset_weight_k = tl.arange(0, BLOCK_SIZE_K)

    # pointers to the current block of input and output
    x_ptrs = x + offset_x_m[:, None] * stride_x_m + offset_x_k[None, :] * stride_x_k
    weight_ptrs = (
        qweight
        + offset_weight_k[:, None] * stride_qweight_k
