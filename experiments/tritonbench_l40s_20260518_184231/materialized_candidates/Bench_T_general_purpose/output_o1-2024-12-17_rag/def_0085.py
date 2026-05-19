import torch
import triton
import triton.language as tl

# ---------------
# Triton MatMul Kernel
# ---------------
@triton.jit
def _matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_an,
    stride_bm, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    # Program IDs for a 2D grid of blocks
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Block start for M and N
    rm = pid_m * BLOCK_M
    rn = pid_n * BLOCK_N

    # Offsets for each thread within a block
    offs_m = rm + tl.arange(0, BLOCK_M)
    offs_n = rn + tl.arange(0, BLOCK_N)
    # Create a pointer for C
    # We'll accumulate partial sums in 'acc'
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32 if C_ptr.dtype in (tl.float16, tl.float32) else tl.float64)

    # K-dimension loop
    # Each iteration handles a slice of size BLOCK_K
    for k_start in range(0, K, BLOCK_K):
        colsA = k_start + tl.arange(0, BLOCK_K)
        rowsB = k_start + tl.arange(0, BLOCK_K)
        # Compute pointers for A and B
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + colsA[None, :] * stride_an)
        b_ptrs = B_ptr + (rowsB[:, None] * stride_bm + offs_n[None, :] * stride_bn)
        # Load blocks from A and B
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (colsA[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(rowsB[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        # Accumulate
        acc += tl.dot(a, b)

    # Write to C
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

def triton_matmul(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor = None) -> torch.Tensor:
    """
    Performs matrix multiplication using the Triton kernel.
    """
    # Shapes
    M, K = a.shape
    Kb, N = b.shape
    assert K == Kb, "Incompatible matrix dimensions."
    if out is None:
        out = torch.empty((M, N), dtype=torch.result_type(a, b), device=a.device)

    # Strides
    stride_am, stride_an = a.stride()
    stride_bm, stride_bn = b.stride()
    stride_cm, stride_cn = out.stride()

    # Launch config
    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_K = 32

    grid = ( (M + BLOCK_M - 1) // BLOCK_M, (N + BLOCK_N - 1) // BLOCK_N )

    _matmul_kernel
