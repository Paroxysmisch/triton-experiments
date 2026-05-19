import triton
import numpy as np

@triton.jit
def kernel(M_ptr, Out_ptr, matrix_stridex, matrix_stridey, out_stridex, out_stridey, SIZE_M, D_HEAD):
    x = triton.program.idx.x
    y = triton.program.idx.y

    # Calculate the pointers for elements of M and Out
    M_ptr += y * matrix_stridey + x * matrix_stridex
    Out_ptr += x * out_stridex + y * out_stridey

    # Load elements from M and store them into Out in a transposed layout
    M_val = triton.program.load(M_ptr, 'float16')
    triton.program.store(Out_ptr, M_val)

def wrapper(SIZE_M, D_HEAD):
    # Initialize matrices
    matrix = np.random.rand(SIZE_M, SIZE_M).astype(np.float16)
    out = np.zeros((SIZE_M, SIZE_M)).astype(np.float16)

    # Copy matrices to GPU
    matrix_gpu = triton.gpu.array(matrix)
    out_gpu = triton.gpu.array(out)

    # Define grid configuration
    grid = (1, )

    # Call the kernel
    kernel[grid](matrix_gpu.ptr, out_gpu.ptr, SIZE_M, D_HEAD, SIZE_M, SIZE_M, SIZE_M, D_HEAD)

    # Copy the result back to CPU
    out = out_gpu.numpy()

    return out
