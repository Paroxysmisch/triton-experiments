import triton
import triton.language as tl

@triton.jit
def matmul4_kernel(
    A, B, C, scales, zeros,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    # Compute block indices
    pid = tl.program_id(axis=0)
    num_blocks_m = M // BLOCK_M
    num_blocks_n = N // BLOCK_N
    block_m = pid // num_blocks_n
    block_n = pid % num_blocks_n

    # Define offsets
    offs_am = block_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = block_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers to the blocks
    A_block_ptr = A + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    B_block_ptr = B + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float16)

    # Loop over K dimension
    for k in range(0, K, BLOCK_K):
        # Load A and B blocks
        A_block = tl.load(A_block_ptr)
        B_block = tl.load(B_block_ptr)

        # Dequantize B
        B_block = (B_block.to(tl.float16) - zeros) * scales

        # Matrix multiplication
        acc += tl.dot(A_block, B_block)

        # Update pointers
        A_block_ptr += BLOCK_K * stride_ak
        B_block_ptr += BLOCK_K * stride_bk

    # Store the result
    C_block_ptr = C + offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn
    tl.store(C_block_ptr, acc)


def matmul_dequantize_int4_gptq(x, qweight, scales, qzeros, output=None):
    M, K = x.shape
    _, N = qweight.shape
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32

    # Initialize output if not provided
    if output is None:
        output = torch.empty((M, N), dtype=torch.float16, device=x.device)

    # Calculate grid size
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)

    # Launch kernel
    matmul4_kernel[grid](
        x, qweight, output, scales, qzeros,
        M, N, K,
        x.stride(0), x.stride(1),
        qweight.stride(0), qweight.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
    )

    return output
