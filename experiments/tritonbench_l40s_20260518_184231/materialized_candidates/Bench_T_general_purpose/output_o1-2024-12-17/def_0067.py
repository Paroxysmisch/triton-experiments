import torch
import torch.nn.functional as F
import triton
import triton.language as tl

@triton.jit
def _pairwise_distance_kernel(
    x1_ptr, x2_ptr, out_ptr,
    batch, stride,
    p, eps,
    BLOCK_SIZE
