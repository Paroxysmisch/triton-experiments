import triton
import triton.language as tl
import torch

@triton.jit
def kernel(M, Out, matrix_stridex, matrix_stridey, out_stridex, out_stridey, SIZE_M, D_HEAD):
    # Calculate the row and column indices for the block
    pid = tl.program_id(0)
    # Assuming the grid size is one-dimensional
    row_idx = pid * SIZE_M + tl.arange(0, SIZE_M)
    col_idx = tl.arange(0, D_HEAD)
    
    # Compute pointers for the input matrix M
    m_ptrs = M + row_idx[:, None] * matrix_stridex + col_idx[None, :] * matrix_stridey
    # Compute pointers for the output matrix Out
    out_ptrs = Out + col_idx[:, None] * out_stridex + row_idx[None, :] * out_stridey
    
    # Load elements from M
    m_vals = tl.load(m_ptrs)
    # Store elements into Out, transposed
    tl.store(out_ptrs, m_vals)

def wrapper(SIZE_M, D_HEAD):
    # Initialize the input matrix with random float16 values
    matrix = torch.randn((SIZE_M, D_HEAD), dtype=torch.float16, device='cuda')
    # Initialize the output matrix with zeros
    out = torch.zeros((D_HEAD, SIZE_M), dtype=torch.float16, device='cuda')
    
    # Define the strides for input and output matrices
    matrix_stridex, matrix_stridey = matrix.stride()
    out_stridex, out_stridey = out.stride()
    
    # Define grid configuration
    grid = (SIZE_M, )  # Assuming a 1D grid based on the number of rows in M
    
    # Launch the kernel
    kernel[grid](
        matrix, out,
        matrix_stridex, matrix_stridey,
        out_stridex, out_stridey,
        SIZE_M, D_HEAD
    )
    
    # Return the transposed matrix
    return out

# Example usage
SIZE_M = 128  # Number of rows in the input matrix
D_HEAD = 64   # Number of columns in the input matrix
transposed_matrix = wrapper(SIZE_M, D_HEAD)
print(transposed_matrix)
