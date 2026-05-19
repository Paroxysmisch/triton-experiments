import triton
import triton.language as tl
import torch
import numpy as np

@triton.jit
def kernel(M, Out, stride_matrix_x, stride_matrix_y, stride_out_x, stride_out_y, SIZE_M: tl.constexpr, D_HEAD: tl.constexpr):
    pid = tl.program_id(0)
    # This implementation uses simple pointer arithmetic to switch the x and y
    # dimensions of the input, which is equivalent to a transpose operation.
    # matrix_idx = tl.arange(0, D_HEAD)
    # The indices for the input are laid out in memory as a 1D vector, so we need to
    # figure out the "x" and "y" indices for a given flat index. 
    # The linear index of a 2D tensor is given by the formula: 
    # index = y * stride_y + x, which in 1D representation is: index = x + width * y
    # matrix_x = matrix_idx % D_HEAD
    # matrix_y = matrix_idx // D_HEAD 

    matrix_x = pid % SIZE_M
    matrix_y = pid // SIZE_M

    # This calculates the linar index "index" of the 1D representation. 
    matrix_idx = matrix_y * stride_matrix_y + matrix_x

    # Fetch the value from the input matrix into the output matrix using 1D indexing.
    matrix_value = tl.load(M + matrix_idx)
    # tl.debug_barrier()
    # Set the transposed value into out.
    out_x = matrix_y
    out_y = matrix_x * D_HEAD
    out_idx = out_x * stride_out_x + out_y

    tl.store(Out + out_idx, matrix_value)


def wrapper(ftest, size_m, d_head, CHECK):
    if CHECK == 1:
        check_precision = 'tf32'
    else:
        check_precision = 'none'

    global precisions_enabled
    matrix = torch.randn((size_m * size_m), device="cuda", dtype=torch.float16)
    out = torch.zeros((size_m * d_head, size_m * d_head), device="cuda", dtype=torch.float16)

    # Enqueue kernel.
    grid = (size_m * d_head * d_head,)

    kernel[grid](matrix, out, size_m, d_head, out.stride(0), out.stride(1), size_m, d_head)
    ftest(out)
    return out
