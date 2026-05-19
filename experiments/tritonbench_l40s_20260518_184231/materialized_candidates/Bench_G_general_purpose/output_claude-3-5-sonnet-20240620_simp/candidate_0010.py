import triton
import triton.language as tl
import torch

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 128, 'BLOCK_M': 32}),
        triton.Config({'BLOCK_N': 256, 'BLOCK_M': 32}),
        triton.Config({'BLOCK_N': 64, 'BLOCK_M': 32}),
        triton.Config({'BLOCK_N': 128, 'BLOCK_M': 64}),
    ],
    key=['N', 'M'],
)
@triton.jit
def mv_kernel(
    # Pointers to matrices
    C_ptr, A_ptr, B_ptr,
    # Matrix dimensions
    N, M,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. stride_am is how much to increase A_ptr
    # by to get the element one row down (A has M columns)
    stride_an, stride_am,  
    stride_bm,
    stride_cn,
    # Meta-parameters
    BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr,
):
    """
    Compute matrix-vector multiplication C = A @ B
    A: (N, M) matrix
    B: (M,) vector
    C: (N,) vector
    """
    # Program ID
    pid = tl.program_id(axis=0)
    
    # Calculate the row range this program instance is responsible for
    row_start = pid * BLOCK_N
    row_offsets = row_start + tl.arange(0, BLOCK_N)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    
    # Iterate through the matrix in blocks
    for m in range(0, M, BLOCK_M):
        # Compute the column indices for this block
        col_offsets = m + tl.arange(0, BLOCK_M)
        
        # Create mask for bounds checking
        row_mask = row_offsets < N
        col_mask = col_offsets < M
        
        # Load the block of the matrix A
        a = tl.load(A_ptr + row_offsets[:, None] * stride_an + col_offsets[None, :] * stride_am,
                   mask=row_mask[:, None] & col_mask[None, :])
        
        # Load the block of vector B
        b = tl.load(B_ptr + col_offsets * stride_bm,
                   mask=col_mask)
        
        # Compute matrix-vector product for this block
        acc += tl.sum(a * b[None, :], axis=1)
    
    # Store the result
    tl.store(C_ptr + row_offsets * stride_cn,
             acc,
             mask=row_mask)

def matrix_vector_multiply(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Compute matrix-vector multiplication C = A @ B using Triton kernel
    
    Args:
        A: torch.Tensor of shape (N, M)
        B: torch.Tensor of shape (M,)
    Returns:
        C: torch.Tensor of shape (N,)
    """
    assert A.is_contiguous()
    assert B.is_contiguous()
    assert A.shape[1] == B.shape[0], "Incompatible dimensions"
    
    N, M = A.shape
    
    # Allocate output
    C = torch.empty(N, device=A.device, dtype=A.dtype)
    
    # Define grid
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_N']),)
    
    # Launch kernel
    mv_kernel[grid](
        C, A, B,
        N, M,
        A.stride(0), A.stride(1),
        B.stride(0),
        C.stride(0)
    )
    
    return C
