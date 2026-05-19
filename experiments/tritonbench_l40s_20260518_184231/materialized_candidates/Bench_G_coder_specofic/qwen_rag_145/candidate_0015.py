import numpy as np
import triton.language as tl
import triton

def broadcastable(a_shape, b_shape):
    # Broadcasting rules for handling arrays of different shapes
    if len(a_shape) < len(b_shape):
        a_shape = [1] * (len(b_shape) - len(a_shape)) + list(a_shape)
    elif len(b_shape) < len(a_shape):
        b_shape = [1] * (len(a_shape) - len(b_shape)) + list(b_shape)

    return tuple(max(a, b) for a, b in zip(a_shape, b_shape))

@triton.jit
def masked_select_kernel(inp_ptr, select_mask_ptr, prefix_sum_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Calculate global IDs
    block_start = tl.program_id(0) * BLOCK_SIZE
    thread_ids = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Check if within bounds of n_elements
    mask = thread_ids < n_elements
    
    # Load input and mask values
    input_val = tl.load(inp_ptr + thread_ids, mask)
    mask_val = tl.load(select_mask_ptr + thread_ids, mask)
    
    # Convert mask to Boolean type
    mask_val_bool = tl.where(mask_val > 0, tl.int1(1), tl.int1(0))
    
    # Calculate output positions using prefix sum
    pos = tl.sum(mask_val_bool[:-1])
    pos = pos + tl.load(prefix_sum_ptr + thread_ids)

    # Store selected values
    output_val = tl.where(mask_val > 0, input_val, tl.int32(0))
    tl.store(out_ptr + pos, output_val, mask)


def masked_select(inp, mask, out):
    # Preprocessing steps
    inp, mask, out = np.asarray(inp), np.asarray(mask), np.asarray(out)
    inp_flat = inp.flatten()
    mask_flat = mask.flatten()
    prefix_sum = np.cumsum(mask_flat)
    out_flat = np.zeros_like(inp_flat)

    # Kernel invocation
    n_elements = inp.size
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.next_power_of_2(n_elements) + BLOCK_SIZE - 1) // BLOCK_SIZE
    masked_select_kernel[grid](inp_flat, mask_flat, prefix_sum, out_flat, n_elements, BLOCK_SIZE)

    # Reshape output
    out.reshape(out_flat.shape)
    out[:] = out_flat
