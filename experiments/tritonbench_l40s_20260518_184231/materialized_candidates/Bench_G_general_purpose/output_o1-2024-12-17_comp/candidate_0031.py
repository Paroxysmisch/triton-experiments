import torch
import triton
import triton.language as tl

@triton.jit
def mul_kernel(
    src_ptr,  # pointer to source data
    dst_ptr,  # pointer to destination data
    n_elements,  # total number of elements in src
    BLOCK_SIZE: tl.constexpr  # compile-time constant block size
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(src_ptr + offsets, mask=mask, other=0.0)
    x = x * (2.0 ** (127 - 15))
    tl.store(dst_ptr + offsets, x, mask=mask)

def launch_mul_kernel(src: torch.Tensor, BLOCK_SIZE: int = 1024):
    # Create destination tensor on the same device as src
    dst = torch.empty_like(src)

    # Calculate the total number of elements and grid size
    n_elements = src.numel()
    grid = lambda meta: ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    # Launch the Triton kernel
    mul_kernel[grid](src, dst, n_elements, BLOCK_SIZE)

    return dst
