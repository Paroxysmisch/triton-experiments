import triton
import triton.language as tl
import torch

# Kernel function for matrix transposition
@triton.jit
def kernel(M, Out, matrix_stridex, matrix_stridey, out_stridex, out_stridey, SIZE_M, D_HEAD, BLOCK_SIZE: tl.constexpr):
    # Compute the position of the current thread
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Load the elements from the input matrix
    matrix_offsets = (offsets // D_HEAD) * matrix_stridey + (offsets % D_HEAD) * matrix_stridex
    matrix_elements = tl.load(M + matrix_offsets, mask=offsets < SIZE_M * D_HEAD, other=0.0)

    # Compute the positions in the output matrix
    out_offsets = (offsets % D_HEAD) * out_stridex + (offsets // D_HEAD) * out_stridey

    # Store the elements in the output matrix
    tl.store(Out + out_offsets, matrix_elements, mask=offsets < SIZE_M * D_HEAD)

# Wrapper function to initialize matrices and call the kernel
def wrapper(matrix, out, SIZE_M, D_HEAD, BLOCK_SIZE=128):
    # Initialize the matrix with random float16 values
    matrix = torch.rand((SIZE_M, D_HEAD), dtype=torch.float16, device='cuda')
    # Initialize the output buffer with zeros
    out = torch.zeros((D_HEAD, SIZE_M), dtype=torch.float16, device='cuda')

    # Define the grid configuration
    grid = (triton.cdiv(SIZE_M * D_HEAD, BLOCK_SIZE),)

    # Call the kernel
    kernel[grid](matrix, out, matrix.stride(1), matrix.stride(0), out.stride(1), out.stride(0), SIZE_M, D_HEAD, BLOCK_SIZE)

    return out

# Example usage
SIZE_M = 1024
D_HEAD = 512
transposed_matrix = wrapper(None, None, SIZE_M, D_HEAD)
print(transposed_matrix)
