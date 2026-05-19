import triton.language as tl
import torch

@triton.jit
def matmul_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    M,
    N,
    K,
    ldA,
    ldB,
    ldC,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    num_stages: tl.constexpr,
    num_warps: tl.constexpr,
    use_wmma: tl.constexpr,
):
    # Indices for thread execution
    m_row = tl.program_id(axis=0)
    n_col = tl.program_id(axis=1)
    k = tl.program_id(axis=2)

    # Compute offsets
    offsets_A = m_row * ldA + k * BLOCK_SIZE_K
    offsets_B = k * ldB + n_col * BLOCK_SIZE_N
    offsets_C = m_row * ldC + n_col * BLOCK_SIZE_N

    # Initialize local accumulator
    local_accumulator = 0.0

    # Load data into registers
    for stage in range(num_stages):
        mask = (stage < num_stages - 1)
        A = tl.load(A_ptr + offsets_A + stage * BLOCK_SIZE_K, mask=mask)
        B = tl.load(B_ptr + offsets_B + stage * BLOCK_SIZE_N, mask=mask)

        # Compute dot product and accumulate to local accumulator
        local_accumulator += tl.dot(A, B)

    # Write results to output
    C = local_accumulator
    tl.store(C_ptr + offsets_C, C)

def matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    BLOCK_SIZE_M: int,
    num_stages: int,
    num_warps: int,
    use_wmma: bool,
    device: torch.device,
):
    # Prepare inputs and meta-parameters
    M, K = A.shape
    N, _ = B.shape
    assert C.shape == (M, N)

    # Ensure input compatibility
    assert A.dtype == B.dtype == C.dtype
    assert A.device == B.device == C.device

    # Establish execution grid dimensions
    grid = (
        tl.grid(M, N, BLOCK_SIZE_M),
        tl.grid(M, N, BLOCK_SIZE_N),
        tl.grid(M, N, BLOCK_SIZE_K),
    )

    # Set device memory for output
    C = C.to(device)

    # Run the kernel
    matmul_kernel[grid](
        A.contiguous().data_ptr(),
        B.contiguous().data_ptr(),
        C.data_ptr(),
        M,
        N,
        K,
        A.stride(0),
        B.stride(0),
        C.stride(0),
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
        num_stages,
        num_warps,
        use_wmma,
    )

    # Synchronize to ensure all threads have completed before returning
    tl.sync()
