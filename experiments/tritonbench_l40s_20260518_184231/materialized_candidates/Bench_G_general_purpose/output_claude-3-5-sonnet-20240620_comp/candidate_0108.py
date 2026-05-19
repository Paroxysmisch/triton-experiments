import triton
import triton.language as tl
import torch

@triton.jit
def kernel(
    M, Out,
    matrix_stridex, matrix_stridey,
    out_stridex, out_stridey,
    SIZE_M, D_HEAD,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate the block start indices
    block_start_m = (pid // (D_HEAD // BLOCK_SIZE_N)) * BLOCK_SIZE_M
    block_start_n = (pid % (D_HEAD // BLOCK_SIZE_N)) * BLOCK_SIZE_N

    # Define offsets for the block
    offs_m = block_start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_n = block_start_n + tl.arange(0, BLOCK_SIZE_N)
    
    # Create a mask for bounds checking
    mask_m = offs_m < SIZE_M
    mask_n = offs_n < D_HEAD
    
    # Compute input and output pointers
    m_ptrs = M + offs_m[:, None] * matrix_stridex + offs_n[None, :] * matrix_stridey
    out_ptrs = Out + offs_n[:, None] * out_stridex + offs_m[None, :] * out_stridey
    
    # Load the input matrix block
    matrix = tl.load(m_ptrs, mask=mask_m[:, None] & mask_n[None, :])
    
    # Transpose the block
    transposed = tl.trans(matrix)
    
    # Store the transposed block
    tl.store(out_ptrs, transposed, mask=mask_n[:, None] & mask_m[None, :])

def wrapper(size_m, d_head):
    # Initialize input matrix with random values
    matrix = torch.randn((size_m, d_head), dtype=torch.float16, device='cuda')
    # Initialize output matrix with zeros
    out = torch.zeros((d_head, size_m), dtype=torch.float16, device='cuda')
    
    # Define block sizes
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    
    # Calculate grid size
    grid = ((size_m + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M) * ((d_head + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N),
    
    # Get matrix strides
    matrix_stridex = matrix.stride(0)
    matrix_stridey = matrix.stride(1)
    out_stridex = out.stride(0)
    out_stridey = out.stride(1)
    
    # Launch kernel
    kernel[grid](
        matrix, out,
        matrix_stridex, matrix_stridey,
        out_stridex, out_stridey,
        size_m, d_head,
        BLOCK_SIZE_M, BLOCK_SIZE_N
    )
    
    return out
