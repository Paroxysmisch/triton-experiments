import triton
import triton.language as tl

# Define the block size
BLOCK_SIZE = 256

# Triton kernel for masked select operation
@triton.jit
def masked_select_kernel(
    inp_ptr: tl.tensor,  # Input data
    select_mask_ptr: tl.tensor,  # Selection mask
    prefix_sum_ptr: tl.tensor,  # Prefix sum array
    out_ptr: tl.tensor,  # Output buffer
    n_elements: tl.int32  # Number of elements
):
    # Compute global ID
    pid = tl.program_id(axis=0)
    coords = tl.arange(0, n_elements, BLOCK_SIZE)
    coords = coords + pid * BLOCK_SIZE

    # Load input and mask values
    inp = tl.load(inp_ptr + coords)
    select_mask = tl.load(select_mask_ptr + coords)

    # Convert mask to Boolean type
    select_mask = select_mask > 0

    # Calculate output positions using prefix sum
    out_pos = tl.load(prefix_sum_ptr + coords)

    # Store selected input values in the output buffer
    tl.store(out_ptr + out_pos, inp, mask=select_mask)

# Wrapper function for masked select operation
def masked_select(
    inp: tl.tensor,  # Input data
    select_mask: tl.tensor,  # Selection mask
):
    # Ensure input and mask are broadcastable
    assert broadcastable(inp.shape, select_mask.shape), "Input and mask shapes are not broadcastable"

    # Flatten the mask if it's not 1D
    if len(select_mask.shape) > 1:
        select_mask = select_mask.flatten()

    # Compute prefix sums for determining output positions
    prefix_sum = tl.cumsum(select_mask)

    # Determine output size
    out_size = prefix_sum[-1]

    # Allocate output buffer
    out = tl.zeros((out_size,), dtype=inp.dtype)

    # Compute grid size
    grid_size = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch kernel
    masked_select_kernel[grid_size, BLOCK_SIZE](inp, select_mask, prefix_sum, out, n_elements)

    return out

# Function to check if two shapes are broadcastable
def broadcastable(shape1, shape2):
    shape1 = shape1[::-1]
    shape2 = shape2[::-1]
    max_shape = max(len(shape1), len(shape2))
    shape1 = [1] * (max_shape - len(shape1)) + shape1
    shape2 = [1] * (max_shape - len(shape2)) + shape2

    for s1, s2 in zip(shape1, shape2):
        if s1 != s2 and s1 != 1 and s2 != 1:
            return False
    return True

# Function to generate various Triton configurations for autotuning
def cfggen():
    configs = []
    for block_size in [128, 256, 512]:
        for warp_size in [1, 2, 4]:
            configs.append(triton.Config({'BLOCK_SIZE': block_size, 'WARP_SIZE': warp_size}))
    return configs
