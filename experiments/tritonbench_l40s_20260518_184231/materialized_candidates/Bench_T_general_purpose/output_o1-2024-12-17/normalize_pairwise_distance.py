import triton
import triton.language as tl
import torch

@triton.jit
def _pairwise_distance_kernel(
    x1_ptr, x2_ptr, out_ptr,
    n_x1, n_x2, d,
    stride_x1n, stride_x1d,
    stride_x2n, stride_x2d,
    stride_outn, stride_outm,
    p_distance: tl.constexpr, eps_distance: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    """
    Kernel to compute pairwise distance between x1 and x2. Each block computes
    a submatrix of the output. The final distance is raised to (1/p_distance),
    then eps_distance is added.
    """
    row_idx = tl.program_id(0)
    col_idx = tl.program_id(1)

    row_start = row_idx * BLOCK_SIZE_M
    col_start = col_idx * BLOCK_SIZE_N

    # Create a 2D range of offsets for the output tile
    offsets_m = row_start + tl.arange(0, BLOCK_SIZE_M)
    offsets_n = col_start + tl.arange(0, BLOCK_SIZE_N)

    # Create an accumulator for partial sums
    acc = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)

    # Loop over the dimension d to compute pairwise distance
    # (abs(x1[i, k] - x2[j, k])^p_distance)
    for k in range(0, d):
        x1_val = tl.load(
            x1_ptr + offsets_m[:, None] * stride_x1n + k * stride_x1d,
            mask=(offsets_m[:, None] < n_x1),
            other=0.0
        )
        x2_val = tl.load(
            x2_ptr + offsets_n
