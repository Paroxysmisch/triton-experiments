import triton
import triton.language as tl

@triton.jit
def masked_select_kernel(
    inp_ptr,  # Pointer to the input data
    select_mask_ptr,  # Pointer to the selection mask
    prefix_sum_ptr,  # Pointer to the prefix sum array
    out_ptr,  # Pointer to the output buffer
    n_elements,  # Number of elements in the input
    BLOCK_SIZE: tl.constexpr  # Block size
):
    pid = tl.program_id(axis=0)  # Get the program ID
    block_start = pid * BLOCK_SIZE  # Calculate the start index for this block

    offsets = block_start + tl.arange(0, BLOCK_SIZE)  # Generate offsets for this block
    mask = offsets < n_elements  # Create a mask to ensure we don't go out of bounds

    # Load input and mask values
    inp = tl.load(inp_ptr + offsets, mask=mask)
    select_mask = tl.load(select_mask_ptr + offsets, mask=mask)

    # Convert the mask to a boolean type
    select_mask = select_mask.to(tl.int1)

    # Calculate output positions using the prefix sum
    output_offsets = tl.load(prefix_sum_ptr + offsets, mask=mask)

    # Store the selected input values in the output buffer
    tl.store(out_ptr + output_offsets, inp, mask=select_mask & mask)

import torch
import triton
import triton.runtime

def broadcastable(shape1, shape2):
    if len(shape1) < len(shape2):
        shape1, shape2 = shape2, shape1
    for a, b in zip(reversed(shape1), reversed(shape2)):
        if a != b and a != 1 and b != 1:
            return False
    return True

def masked_select(inp, select_mask):
    # Ensure the input and mask are broadcastable
    if not broadcastable(inp.shape, select_mask.shape):
        raise ValueError("Input and mask are not broadcastable")

    # Broadcast tensors to compatible shapes
    inp, select_mask = torch.broadcast_tensors(inp, select_mask)

    # Flatten the mask
    select_mask = select_mask.flatten()

    # Compute the prefix sum of the mask
    prefix_sum = select_mask.cumsum(dim=0).int()

    # Determine the number of elements
    n_elements = select_mask.numel()

    # Determine the number of selected elements
    n_selected = prefix_sum[-1].item()

    # Allocate the output buffer
    out = torch.empty(n_selected, dtype=inp.dtype, device=inp.device)

    # Define the block size
    BLOCK_SIZE = 128

    # Compute the grid size
    grid_size = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch the kernel
    masked_select_kernel[grid_size, BLOCK_SIZE](
        inp.flatten().contiguous().ptr,
        select_mask.contiguous().ptr,
        prefix_sum.contiguous().ptr,
        out.contiguous().ptr,
        n_elements
    )

    return out

def cfggen():
    # Define a range of block sizes and warp numbers to test
    block_sizes = [128, 256, 512, 1024]
    warp_numbers = [1, 2, 4, 8]

    # Generate configurations
    configs = []
    for block_size in block_sizes:
        for warp_number in warp_numbers:
            configs.append({
                'BLOCK_SIZE': block_size,
                'WARP_NUMBER': warp_number
            })

    return configs
