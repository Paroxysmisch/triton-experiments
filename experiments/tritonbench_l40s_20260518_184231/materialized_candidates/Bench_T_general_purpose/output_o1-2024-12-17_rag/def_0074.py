import torch
import triton
import triton.language as tl
from torch import Tensor
from triton.runtime.jit import get_cuda_stream

@triton.jit
def _normalized_cosine_similarity_kernel(
    x1_ptr, 
    x2_ptr, 
    out_ptr, 
    stride, 
    n_cols, 
    eps_norm, 
    eps_similarity, 
    p_norm,
    N_COLS: tl.constexpr, 
    BLOCK_SIZE: tl.constexpr
):
    row_id = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    row_start_x1 = x1_ptr + row_id * stride
    row_start_x2 = x2_ptr + row_id * stride

    # 1. Compute p-norm of x1 and x2
    p_sum_x1 = tl.float32(0)
    p_sum_x2 = tl.float32(0)
    for start_col in range(0, N_COLS, BLOCK_SIZE):
        ptr_x1 = row_start_x1 + start_col + offsets
        ptr_x2 = row_start_x2 + start_col + offsets
        mask = (start_col + offsets) < n_cols

        x1 = tl.load(ptr_x1, mask=mask, other=0.).to(tl.float32)
        x2 = tl.load(ptr_x2, mask=mask, other=0.).to(tl.float32)

        x1_p = tl.abs(x1) ** p_norm
        x2_p = tl.abs(x2) ** p_norm
        p_sum_x1 += tl.sum(x1_p, axis=0)
        p_sum_x2 += tl.sum(x2_p, axis=0)

    norm_x1 = tl.pow
