import triton
import triton.language as tl
import torch

@triton.jit
def mv_kernel(
    # Pointers to matrices/vectors
    A_ptr, B_ptr, C_ptr,
    # Matrix dimensions
    N, M,
    # Stride information
    stride_am, stride_an,
    stride_bm,
    stride_cn,
    # Meta-parameters
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Matrix-vector multiplication kernel.
    
    A: shape (N, M) in row-major layout
    B: shape (M,) 
    C: shape (N,) (output)
    """
    # Get program ID for row blocks
    pid = tl.program_id(0)
    row_start = pid * BLOCK_N
    rows = row_start + tl.arange(0, BLOCK_N)
    row_mask = rows < N

    # Initialize accumulator
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # Loop over column blocks
    for col_start in range(0, M, BLOCK_M):
        cols = col_start + tl.arange(0, BLOCK_M)
        col_mask = cols < M

        # Load A block with appropriate masking
        a_ptrs = A_ptr + rows[:, None] * stride_an + cols[None, :] * stride_am
        a = tl.load(a_ptrs, mask=row_mask[:, None] & col_mask[None, :], other=0.0)
        
        # Load B vector block
        b_ptrs = B_ptr + cols * stride_bm
        b = tl.load(b_ptrs, mask=col_mask, other=0.0)

        # Accumulate partial sums with FP32 precision
        acc += tl.sum(a.to(tl.float32) * b.to(tl.float32), axis=1)

    # Write back result with appropriate masking
    c_ptrs = C_ptr + rows * stride_cn
    tl.store(c_ptrs, acc.to(tl.float32), mask=row_mask)

def mv(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    BLOCK_N: int = 128,
    BLOCK_M: int = 64
):
    """Matrix-vector multiplication wrapper.
    
    Args:
        A: Matrix of shape (N, M)
        B: Vector of shape (M,)
        C: Output vector of shape (N,)
        BLOCK_N: Number of rows processed per block
        BLOCK_M: Number of columns processed per iteration
    """
    # Input validation
    assert A.is_cuda and B.is_cuda and C.is_cuda, "Tensors must be on GPU"
    assert A.dim() == 2 and B.dim() == 1 and C.dim() == 1, "Invalid tensor dimensions"
    N, M = A.shape
    assert B.shape[0] == M, "Dimension mismatch between A and B"
    assert C.shape[0] == N, "Dimension mismatch between A and C"

    # Compute execution grid
    grid = (triton.cdiv(N, BLOCK_N),)

    # Launch kernel
    mv_kernel[grid](
        A, B, C,
        N, M,
        A.stride(1), A.stride(0),  # Strides for A (row-major)
        B.stride(0),               # Strides for B
        C.stride(0),               # Strides for C
        BLOCK_N=BLOCK_N,
        BLOCK_M=BLOCK_M,
    )

# Example usage:
N, M = 1024, 512
A = torch.rand(N, M, device='cuda', dtype=torch.float32)
B = torch.rand(M, device='cuda', dtype=torch.float32)
C = torch.empty(N, device='cuda', dtype=torch.float32)

mv(A, B, C, BLOCK_N=128, BLOCK_M=64)
