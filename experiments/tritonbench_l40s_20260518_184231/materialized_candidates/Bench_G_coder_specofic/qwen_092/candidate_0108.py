import triton
import triton.language as tl

@triton.jit
def kernel(M_ptr, Out_ptr, matrix_stridex, matrix_stridey, out_stridex, out_stridey, SIZE_M, D_HEAD):
    # Get the 2D index of the current thread
    pid = tl.program_id(0)
    
    # Compute the row and column indices in the original matrix
    row = pid % SIZE_M
    col = pid // SIZE_M
    
    # Compute the pointers for the current element in M and Out
    M_addr = M_ptr + row * matrix_stridex + col * matrix_stridey
    Out_addr = Out_ptr + col * out_stridex + row * out_stridey
    
    # Load the element from M
    M_val = tl.load(M_addr)
    
    # Store the element in Out
    tl.store(Out_addr, M_val)

@triton.jit
def wrapper(SIZE_M, D_HEAD):
    # Initialize the random seed
    tl.random.seed(1234)
    
    # Allocate memory for the input matrix M and the output matrix Out
    matrix = tl.zeros((SIZE_M * D_HEAD,), dtype=tl.float16)
    out = tl.zeros((SIZE_M * D_HEAD,), dtype=tl.float16)
    
    # Fill the input matrix M with random float16 values
    for i in range(SIZE_M * D_HEAD):
        matrix[i] = tl.random.rand()
    
    # Define the strides for the input and output matrices
    matrix_stridex = SIZE_M * D_HEAD
    matrix_stridey = 1
    out_stridex = 1
    out_stridey = SIZE_M * D_HEAD
    
    # Define the grid configuration
    grid = (SIZE_M * D_HEAD,)
    
    # Call the kernel to transpose the matrix
    kernel[grid](matrix, out, matrix_stridex, matrix_stridey, out_stridex, out_stridey, SIZE_M, D_HEAD)
    
    # Return the transposed matrix
    return out

# Example usage
SIZE_M = 16
D_HEAD = 16
transposed_matrix = wrapper(SIZE_M, D_HEAD)
print(transposed_matrix)
