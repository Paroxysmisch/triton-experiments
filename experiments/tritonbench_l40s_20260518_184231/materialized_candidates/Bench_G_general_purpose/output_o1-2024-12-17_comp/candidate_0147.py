import triton
import triton.language as tl
import torch


@triton.jit
def matmul_tma_load_store(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    lda, ldb, ldc,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    TRANS_A: tl.constexpr, TRANS_B: tl.constexpr,
    OUTPUT_F16: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    m_offset = pid_m * BLOCK_M
    n_offset = pid_n * BLOCK_N

    # Create block pointers for A
    if TRANS_A:
        # A is KxM if transposed
        a_block_ptr = tl.make_block_ptr(
            base=A_ptr,
            shape=(K, M),
            strides=(lda, 1),
            offsets=(0, m_offset),
            block_shape=(BLOCK_K, BLOCK_M),
            order=(0, 1),
        )
    else:
        # A is MxK
        a_block_ptr = tl.make_block_ptr(
            base=A_ptr,
            shape=(M, K),
            strides=(lda, 1),
            offsets=(m_offset, 0),
            block_shape=(BLOCK_M, BLOCK_K),
            order=(0, 1),
        )

    # Create block pointers for B
    if TRANS_B:
        # B is NxK if transposed
        b_block_ptr = tl.make_block_ptr(
            base=B_ptr,
            shape=(N, K),
            strides=(ldb, 1),
            offsets=(n_offset, 0),
            block_shape=(BLOCK_N, BLOCK_K),
            order=(0, 1),
        )
    else:
        # B is KxN
        b_block_ptr = tl.make_block_ptr(
            base=B_ptr,
            shape=(K, N),
            strides=(ldb, 1),
            offsets=(0, n_offset),
            block_shape=(BLOCK_K, BLOCK_N),
            order=(0, 1),
        )

    c_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Block-k loop
    for k_block_start in range(0, K, BLOCK_K):
        # Load sub-block of A
        a = tl.load(a_block_ptr, boundary_check=(0, 1))
        # Load sub-block of B
        b = tl.load(b_block_ptr, boundary_check=(0, 1))
        # Update pointers to move to next block (only if within bounds)
        if TRANS_A:
            a_block_ptr = tl.advance8x4(a_block_ptr, (BLOCK_K, 0))
        else:
            a_block_ptr = tl.advance8x4(a_block_ptr, (0, BLOCK_K))
        if TRANS_B:
            b_block_ptr = tl.advance8x4(b_block_ptr, (0, BLOCK_K))
        else:
            b_block_ptr = tl.advance8x4(b_block_ptr, (BLOCK_K, 0))
        # Accumulate
        c_acc += tl.dot(a, b)

    # Store the result
    # clamp indices
    m_mask = m_offset + tl.arange(0, BLOCK_M) < M
    n_mask = n_offset + tl.arange(0, BLOCK_N) < N
    c_block_ptr = tl.make_block_ptr(
        base=C_ptr,
        shape=(M, N),
        strides=(ldc, 1),
        offsets=(m_offset, n_offset),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(0, 1),
    )
    if OUTPUT_F16:
        c_acc_fp16 = c_acc.to(tl.float16)
        tl.store(c_block_ptr, c_acc_fp16, mask=m_mask[:, None] & n_mask[None, :])
    else:
        tl.store(c_block_ptr, c_acc, mask=m_mask[:, None] & n_mask[None, :])


def warpper_tma_load_store(
    M, N, K,
    BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
    num_warps=4, num_ctas=None,
    trans_a=False, trans_b=False, output_f16=False
):
    # Generate input matrices
    A_shape = (M, K) if not trans_a else (K, M)
    B_shape = (K, N) if not trans_b else (N, K)

    A = torch.randn(A_shape, dtype=torch.float16, device='cuda')
    B = torch.randn(B_shape, dtype=torch.float16, device='cuda')
    C = torch.zeros((M, N), dtype=torch.float16 if output_f16 else torch.float32, device='cuda')

    # Leading dimensions
    lda = A.stride(0)
    ldb = B.stride(0)
    ldc = C.stride(0)

    # Grid
    if num_ctas is None:
        grid = ((M + BLOCK_M - 1) // BLOCK_M, (N + BLOCK_N - 1) // BLOCK_N)
    else:
        grid = num_ctas

    # Launch kernel
    matmul_tma_load_store[grid](
        A, B, C,
        M, N, K,
        lda, ldb, ldc,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        TRANS_A=trans_a,
        TRANS_B=trans_b,
        OUTPUT_F16=output_f16,
        num_warps=num_warps,
        num_stages=2,
    )

    return A, B, C
