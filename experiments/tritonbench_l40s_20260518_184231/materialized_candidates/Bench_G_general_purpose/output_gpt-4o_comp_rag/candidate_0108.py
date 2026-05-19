import triton
import triton.language as tl

@triton.jit
def kernel(M, Out, stride_mx, stride_my, stride_ox, stride_oy, SIZE_M, SIZE_N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    row = pid // SIZE_N
    col = pid % SIZE_N

    # Compute the pointers
    m_ptr = M + row * stride_mx + col * stride_my
    o_ptr = Out + col * stride_ox + row * stride_oy

    # Load and store the elements
    m_val = tl.load(m_ptr)
    tl.store(o_ptr, m_val)

import torch

def wrapper(SIZE_M, SIZE_N, BLOCK_SIZE=16):
    # Initialize the input matrix with random float16 values
    matrix = torch.randn((SIZE_M, SIZE_N), dtype=torch.float16, device='cuda')
    # Initialize the output matrix with zeros
    out = torch.zeros((SIZE_N, SIZE_M), dtype=torch.float16, device='cuda')

    # Define the grid configuration
    grid = (SIZE_M * SIZE_N,)

    # Call the kernel function
    kernel[grid](matrix, out, matrix.stride(0), matrix.stride(1), out.stride(0), out.stride(1), SIZE_M, SIZE_N, BLOCK_SIZE)

    return out

SIZE_M = 1024
SIZE_N = 512
transposed_matrix = wrapper(SIZE_M, SIZE_N)
print(transposed_matrix)
