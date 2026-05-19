import triton
import triton.language as tl
import torch

@triton.jit
def kernel(M, Out,
           matrix_stridex, matrix_stridey,
           out_stridex, out_stridey,
           SIZE_M, D_HEAD,
           BLOCK_SIZE_M: tl.constexpr,
           BLOCK_SIZE_N: tl.constexpr):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate block indices
    block_m = pid // (D_HEAD // BLOCK_SIZE_N)
    block_n = pid % (D_HEAD // BLOCK_SIZE_N)
    
    # Calculate offsets
    offs_m = block_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = block_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Create mask for bounds checking
    mask = (offs_m[:, None] < SIZE_M) & (offs_n[None, :] < D_HEAD)
    
    # Calculate input pointers
    m_ptrs = M + offs_m[:, None] * matrix_stridex + offs_n[None, :] * matrix_stridey
    
    # Load input matrix elements
    matrix = tl.load(m_ptrs, mask=mask, other=0.0)
    
    # Calculate output pointers (transposed indices)
    out_ptrs = Out + offs_n[:, None] * out_stridex + offs_m[None, :] * out_stridey
    
    # Store transposed elements
    tl.store(out_ptrs, tl.trans(matrix), mask=tl.trans(mask))

def wrapper(SIZE_M, D_HEAD):
    # Initialize input matrix with random values
    matrix = torch.randn((SIZE_M, D_HEAD), dtype=torch.float16, device='cuda')
    
    # Initialize output matrix with zeros
    out = torch.zeros((D_HEAD, SIZE_M), dtype=torch.float16, device='cuda')
    
    # Define block sizes
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    
    # Calculate grid size
    grid = ((SIZE_M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M) * ((D_HEAD + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N),
    
    # Launch kernel
    kernel[grid](
        matrix, out,
        matrix.stride(0), matrix.stride(1),
        out.stride(0), out.stride(1),
        SIZE_M, D_HEAD,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N
    )
    
    return out
