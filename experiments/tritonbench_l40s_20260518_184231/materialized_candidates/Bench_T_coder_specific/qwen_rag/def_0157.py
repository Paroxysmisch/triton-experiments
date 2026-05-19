import triton
import triton.language as tl
import torch

@triton.jit
def signbit_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    Kernel to compute the sign bit check for each element in the input tensor.

    Parameters:
    - x_ptr: Pointer to the input tensor.
    - output_ptr: Pointer to the output tensor where results will be stored.
    - n_elements: Total number of elements in the tensor.
    - BLOCK_SIZE: The block size for Triton kernel execution.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    output = tl.bitcast(tl.bitcast(x, tl.uint32) >> 31, tl.bool)
    tl.store(output_ptr + offsets, output, mask=mask)
