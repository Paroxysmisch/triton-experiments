import torch
import triton
import triton.language as tl

@triton.jit
def _permute_copy_kernel_4d(
    A_ptr, O_ptr,
    sIn0, sIn1, sIn2, sIn3,
    stIn0, stIn1, stIn2, stIn3,
    sOut0, sOut1, sOut2, sOut3,
    stOut0, stOut1, stOut2, stOut3,
    d0, d1, d2, d3,
    total_elems,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elems

    # Decode flattened index into input coordinates
    idx = offsets
    i_in0 = idx // (sIn1 * sIn2 * sIn3)
    rem = idx % (sIn1 * sIn2 * sIn3)
    i_in1 = rem // (sIn2 * sIn3)
    rem = rem % (sIn2 * sIn3)
    i_in2 = rem // sIn3
    i_in3 = rem % sIn3

    # Compute input offset
    in_offset = i_in0 * stIn0 + i_in1 * stIn1 + i_in2 * stIn2 + i_in3 * stIn3

    # Map input coords to output coords based on dims
    # d0, d1
