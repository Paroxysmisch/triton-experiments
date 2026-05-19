import torch
import triton
import triton.language as tl
import math

# A small Triton kernel to multiply each column of U by the corresponding singular value in S:
@triton.jit
def apply_s_kernel(
    U_ptr,      # Pointer to U (flattened)
    S_ptr,      # Pointer to
