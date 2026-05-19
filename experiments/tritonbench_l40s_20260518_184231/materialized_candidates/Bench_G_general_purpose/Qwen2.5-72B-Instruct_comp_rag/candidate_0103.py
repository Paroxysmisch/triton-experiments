import triton
import triton.language as tl
import torch

@triton.jit
def kernel(M_ptr, Out_ptr, matrix_stridex, matrix_stridey, out_stridex, out_stridey, SIZE_M, D_HEAD):
    """
    Transpose a matrix M of shape (SIZE_M, D_HEAD) and store the result in Out.
    """
    pid = tl.program_id(axis=0)
    block_size = 128  # Define the block size for the kernel
    num_blocks = (SIZE_M + block_size - 1) // block_size
    block_start = pid * block_size
    block_end = min(block_start + block_size, SIZE_M)

    for i in range(block_start, block_end):
        for j in range(D_HEAD):
            # Compute the pointers for the elements in M and Out
            m_ptr = M_ptr + i * matrix_stridex + j * matrix_stridey
            out_ptr = Out_ptr + j * out_stridex + i * out_stridey
            # Load the element from M and store it in Out
            value = tl.load(m_ptr)
            tl.store(out_ptr, value)

def wrapper(SIZE_M, D_HEAD):
    # Initialize the matrix with random float16 values
    matrix = torch.rand((SIZE_M, D_HEAD), dtype=torch.float16, device='cuda')
    # Initialize the output buffer with zeros
    out = torch.zeros((D_HEAD, SIZE_M), dtype=torch.float16, device='cuda')

    # Define the grid configuration
    grid = (triton.cdiv(SIZE_M, 128),)

    # Call the kernel with the matrices and their properties
    kernel[grid](matrix, out, matrix.stride(1), matrix.stride(0), out.stride(1), out.stride(0), SIZE_M, D_HEAD)

    return out

# Example usage
SIZE_M = 1024
D_HEAD = 512
transposed_matrix = wrapper(SIZE_M, D_HEAD)
print(transposed_matrix)
