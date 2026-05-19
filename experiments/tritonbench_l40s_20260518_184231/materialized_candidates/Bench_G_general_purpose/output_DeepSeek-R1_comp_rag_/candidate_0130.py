import triton
import triton.language as tl
import torch

@triton.jit
def kernel_function(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Determine the block's starting index
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    # Generate offsets
