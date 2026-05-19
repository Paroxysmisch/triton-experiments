import torch
import triton
import triton.language as tl

# Define block sizes for efficient memory access
BLOCK_M = 128
BLOCK_N = 128
BLOCK_K = 32

@triton.jit
def matmul_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. stride_am is how much to increase a_ptr
    # by to get the element one row down (A has M rows)
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Meta-parameters
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    Kernel for computing the matrix multiplication C = A x B.
    A has shape (M, K), B has shape (K, N), C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Matrix multiplication main loop
    pid = tl.program_id(axis=0)
    
    # We program in a 2D grid
    grid_m = pid // (N // BLOCK_N)
    grid_n = pid % (N // BLOCK_N)

    # Starting offsets for this program
    offs_am = (grid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (grid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Iterate to compute a block of the C matrix
    for k in range(0, K, BLOCK_K):
        # Load the next block of A and B
        a = tl.load(a_ptr + offs_am[:, None] * stride_am + (k + offs_k[None, :]) * stride_ak)
        b = tl.load(b_ptr + (k + offs_k[:, None]) * stride_bk + offs_bn[None, :] * stride_bn)
        
        # We accumulate along the K dimension
        acc += tl.dot(a, b)
    
    # Store the result
    offs_m = grid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = grid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c = acc
    
    # Write back the block of results
    tl.store(c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, c)

def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Compute the matrix multiplication C = A x B
    A: (M, K) matrix
    B: (K, N) matrix
    Returns: (M, N) matrix
    """
    # Check constraints
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    M, K = a.shape
    K, N = b.shape
    
    # Allocate output
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    # 1D launch kernel where each block gets its own program
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),
    )
    
    # Launch kernel
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    
    return c
