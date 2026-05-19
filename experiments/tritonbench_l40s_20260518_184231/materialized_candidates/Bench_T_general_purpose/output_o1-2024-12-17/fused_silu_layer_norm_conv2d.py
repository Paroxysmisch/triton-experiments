import torch
import torch.nn.functional as F
import triton
import triton.language as tl

@triton.jit
def _ln_compute_stats_kernel(
    output_ptr,  # (N, C, H, W) float32
    mean_ptr,    # (N*C,) float32
    var_ptr,     # (N*C,) float32
    H, W,
    eps,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    # Each program handles a single (n, c) pair
    # Flatten n*c into pid
    #
