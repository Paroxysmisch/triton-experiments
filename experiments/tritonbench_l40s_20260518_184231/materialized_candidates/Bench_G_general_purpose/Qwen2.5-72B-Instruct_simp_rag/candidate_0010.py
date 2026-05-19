import triton
import triton.language as tl

@triton.jit
def masked_select_kernel(
    inp_ptr, mask_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr
):
    # Compute the starting index for this block
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE

    # Compute the range of indices for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Create a mask to filter out out-of-bounds indices
    in_bounds = offsets < n_elements

    # Load input and mask data
    inp = tl.load(inp_ptr + offsets, mask=in_bounds)
    mask = tl.load(mask_ptr + offsets, mask=in_bounds)

    # Compute the number of selected elements in this block
    selected_count = tl.sum(mask, axis=0)

    # Compute the prefix sum to determine the output offsets
    prefix_sum = tl.cumsum(mask, axis=0)

    # Compute the output offsets
    out_offsets = block_start + prefix_sum - 1

    # Store the selected elements in the output buffer
    tl.store(out_ptr + out_offsets, inp, mask=mask)

import triton
import triton.language as tl

def masked_select(inp, mask):
    # Ensure the input and mask are broadcastable
    if not broadcastable(inp.shape, mask.shape):
        raise ValueError("Input and mask shapes are not broadcastable")

    # Flatten the input and mask
    inp_flat = inp.flatten()
    mask_flat = mask.flatten()

    # Determine the number of elements
    n_elements = inp_flat.size

    # Allocate the output buffer
    out_size = mask_flat.sum()
    out = triton.empty((out_size,), dtype=inp.dtype, device=inp.device)

    # Define the block size
    BLOCK_SIZE = 256

    # Define the grid size
    grid_size = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch the kernel
    masked_select_kernel[grid_size, BLOCK_SIZE](
        inp_flat.ptr, mask_flat.ptr, out.ptr, n_elements, BLOCK_SIZE
    )

    return out

def broadcastable(shape1, shape2):
    # Check if two shapes are broadcastable
    for a, b in zip(shape1[::-1], shape2[::-1]):
        if a == 1 or b == 1 or a == b:
            continue
        return False
    return True
