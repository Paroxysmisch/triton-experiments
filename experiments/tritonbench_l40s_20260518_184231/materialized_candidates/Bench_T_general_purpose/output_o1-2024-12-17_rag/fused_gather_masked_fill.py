import torch
import triton
import triton.language as tl

@triton.jit
def _fused_gather_masked_fill_kernel(
    output_ptr,         # [*] float* (out), shape = (n_rows, n_cols)
    input_ptr,          # [*] float* (in ), shape = (n_rows
