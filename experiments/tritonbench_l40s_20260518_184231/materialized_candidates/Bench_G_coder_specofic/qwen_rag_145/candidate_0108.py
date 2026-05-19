import triton
import triton.language as tl
import numpy as np

@triton.jit
def kernel(M_ptr, Out_ptr, SIZE_M, Out_stridex, Out_stridey, D_HEAD, matrix_stridex, matrix_stridey,
           BLOCK_SIZE_X: tl.constexpr, BLOCK_SIZE_Y: tl.constexpr):
    pid_x = tl.program_id(axis=0)
    pid_y = tl.program_id(axis=1)

    M = tl.load(M_ptr + pid_x * matrix_stridex + pid_y * matrix_stridey)
    Out = tl.load(Out_ptr + pid_y * Out_stridex + pid_x * Out_stridey)

    Out += M

    tl.store(Out_ptr + pid_y * Out_stridex + pid_x * Out_stridey, Out)


def wrapper(M, Out):
    # Initialize M and Out on CUDA
    dM = triton.differentiate(M)
    dOut = triton.differentiate(Out)

    # Define grid configuration
    grid = (1, )

    # Call the kernel
    kernel[grid](dM, dOut, M.shape[0], Out.shape[1], Out.strides[0], Out.shape[1], M.strides[0], M.strides[1])

    # Copy the result back to CPU
    triton.copy_to_numpy(Out, dOut)

    return Out
