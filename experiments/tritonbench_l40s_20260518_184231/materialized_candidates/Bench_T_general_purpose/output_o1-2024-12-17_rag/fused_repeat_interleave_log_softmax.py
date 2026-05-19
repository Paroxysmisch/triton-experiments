import triton
import triton.language as tl
import torch

@triton.jit
def _fused_repeat_interleave_log_softmax_sum_kernel(
    X,      # [M, N] flattened input
    R,      #
