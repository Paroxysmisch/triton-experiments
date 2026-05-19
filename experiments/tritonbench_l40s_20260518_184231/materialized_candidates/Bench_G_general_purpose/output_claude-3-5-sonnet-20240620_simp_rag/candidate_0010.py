import triton
import triton.language as tl
import torch

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 128, 'BLOCK_M': 32}, num_stages=3),
        triton.Config({'BLOCK_N': 256, 'BLOCK_M': 32}, num_stages=3),
        triton.Config({'BLOCK_N': 64, 'BLOCK_M': 32}, num_stages=4),
        triton.Config({'BLOCK_N': 128, 'BLOCK_M': 64}, num_stages=4),
    ],
    key=['N', 'M'],
)
@triton.jit
def mv_kernel(
    # Pointers to matrices
    A_ptr, B_ptr, C_ptr,
    # Matrix dimensions
    N, M,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. stride_am is how much to increase a_ptr
    # by to get the element one row down (A has M columns)
    stride_an, stride_am,  
    stride_bm,
    # Meta-parameters
    BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr,
):
    """
    Compute C = A @ B where A is a matrix of shape (N, M) and B is a vector of shape (M,)
    """
    # -----------------------------------------------------------
    # Matrix multiplication kernel
    pid = tl.program_id(0)
    
    # Create block pointers for the current program instance
    # Each program handles a block of rows of size BLOCK_N
    block_start = pid * BLOCK_N
    
    # Create offsets for A matrix
    offs_n = block_start + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    
    # Create a mask to guard against out-of-bounds accesses
    mask_n = offs_n < N
    
    # Initialize the accumulator
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    
    # Iterate through the M dimension by BLOCK_M steps
    for m in range(0, M, BLOCK_M):
        # Create masks for bounds checking
        mask_m = offs_m + m < M
        
        # Compute the pointers to load from A and B
        a_ptrs = A_ptr + offs_n[:, None] * stride_an + (offs_m[None, :] + m) * stride_am
        b_ptrs = B_ptr + (offs_m + m) * stride_bm
        
        # Load data from A and B, using masks
        a = tl.load(a_ptrs, mask=mask_n[:, None] & mask_m[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=mask_m, other=0.0)
        
        # Perform matrix-vector multiplication for this block
        acc += tl.sum(a * b[None, :], axis=1)
    
    # Store the results back to C
    c_ptrs = C_ptr + offs_n
    tl.store(c_ptrs, acc, mask=mask_n)

def matrix_vector_multiply(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Compute C = A @ B where A is a matrix and B is a vector
    """
    assert A.ndim == 2 and B.ndim == 1, "Invalid input dimensions"
    assert A.shape[1] == B.shape[0], "Invalid input shapes"
    
    # Get matrix dimensions
    N, M = A.shape
    
    # Allocate output
    C = torch.empty(N, device=A.device, dtype=A.dtype)
    
    # Launch kernel with appropriate grid size
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_N']),)
    
    mv_kernel[grid](
        A_ptr=A.data_ptr(),
        B_ptr=B.data_ptr(),
        C_ptr=C.data_ptr(),
        N=N, M=M,
        stride_an=A.stride(0),
        stride_am=A.stride(1),
        stride_bm=B.stride(0),
    )
    
    return C
