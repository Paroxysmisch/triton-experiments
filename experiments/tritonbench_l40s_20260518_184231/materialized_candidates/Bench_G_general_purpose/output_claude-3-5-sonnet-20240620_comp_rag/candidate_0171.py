import torch
import triton
import triton.language as tl

@triton.jit
def _l2_norm_bwd_kernel(
    DX,  # pointer to input gradient
    X,   # pointer to input
    DY,  # pointer to output gradient
    stride_x_row,  # stride between rows
    N,   # number of columns
    eps, # epsilon to avoid division by zero
    BLOCK_N: tl.constexpr,  # number of elements per block
):
    # Get the row index
    row = tl.program_id(0)
    
    #
