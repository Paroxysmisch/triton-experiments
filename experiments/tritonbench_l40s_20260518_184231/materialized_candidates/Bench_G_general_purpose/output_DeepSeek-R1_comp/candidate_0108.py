import torch
import triton
import triton.language as tl

@triton.jit
def kernel(
    M, Out,
    matrix_stride_x, matrix_stride_y,
    out_stride_x, out_stride_y,
    SIZE_M, D_HEAD,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < SIZE_M * D_HEAD
    
    # Calculate original matrix indices
    i = offsets // D_HEAD
    j = offsets % D_HEAD
    
    # Compute pointers for input and transposed output
    m_ptr = M + i * matrix_stride_x + j * matrix_stride_y
    out_ptr = Out + j * out_stride_x + i * out_stride_y
    
    # Load and store with mask
    val = tl.load(m_ptr, mask=mask)
    tl.store(out_ptr, val, mask=mask)

def wrapper(SIZE_M: int, D_HEAD: int):
    # Initialize matrices on CUDA
    matrix = torch.randn((SIZE_M, D_HEAD), dtype=torch.float16, device='cuda')
    out = torch.zeros((D_HEAD, SIZE_M), dtype=torch.float16, device='cuda')
    
    # Get stride information
    matrix_stride_x, matrix_stride_y = matrix.stride()
    out_stride_x, out_stride_y = out.stride()
    
    # Configure kernel launch parameters
    BLOCK_SIZE = 128
    grid = (triton.cdiv(SIZE_M * D_HEAD, BLOCK_SIZE),)
    
    # Launch kernel
    kernel[grid](
        M=matrix, Out=out,
        matrix_stride_x=matrix_stride_x, matrix_stride_y=matrix_stride_y,
        out_stride_x=out_stride_x, out_stride_y=out_stride_y,
        SIZE_M=SIZE_M, D_HEAD=D_HEAD,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out
