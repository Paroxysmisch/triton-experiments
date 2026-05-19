import math
import torch
import triton
import triton.language as tl

@triton.jit
def _cos_avg_pool1d_kernel(
    input_ptr,  # float32
    output_ptr,  # float32
    B, C, iW, oW,
    kernel_size,
    stride,
    padding,
    count_include_pad,  # 0 or 1
    BLOCK_SIZE: tl.constexpr
):
    # program_id(0) ranges over batch dimension
    # program_id(1) ranges over channel dimension
    # program_id(2) ranges over blocks along the output width
    b_id = tl.program_id(0)
    c_id = tl.program_id(1)
    block_pos = tl.program_id(2)

    out_offset = block_pos * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = out_offset < oW

    # Base pointers for the current (b_id, c_id)
    in_base = b_id * (C * iW) + c_id * iW
    out_base = b_id * (C * oW) + c_id * oW

    # Initialize accumulators
    sum_vals = tl.zeros_like(out_offset, dtype=tl
