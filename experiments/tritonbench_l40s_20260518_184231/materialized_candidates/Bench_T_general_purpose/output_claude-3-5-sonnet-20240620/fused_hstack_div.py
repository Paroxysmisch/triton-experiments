import triton
import triton.language as tl

@triton.jit
def fused_hstack_div_kernel(tensors_ptr, divisor_ptr, out_ptr, num_tensors, rounding_mode, n_elements):
    # Load tensors and perform horizontal stacking
    stacked_tensor = tl.zeros((num_tensors, n_elements), dtype=tl.float32)  # Adjust dtype as needed
    for i in range(num_tensors):
        stacked_tensor[i] = tl.load(tensors_ptr + i * n_elements)

    # Load divisor
    divisor = tl.load(divisor_ptr)

    # Element-wise division
    result = stacked_tensor / divisor

    # Apply rounding if specified
    if rounding_mode == 1:  # trunc
        result = tl.trunc(result)
    elif rounding_mode == 2:  # floor
        result = tl.floor(result)

    # Store the result
    tl.store(out_ptr, result)

def fused_hstack_div(tensors, divisor, *, rounding_mode=None, out=None):
    # Prepare inputs
    num_tensors = len(tensors)
    n_elements = tensors[0].numel()  # Assuming all tensors have the same number of elements
    tensors_ptr = tl.make_tensor(tensors)
    divisor_ptr = tl.make_tensor(divisor)
    
    # Determine rounding mode
    if rounding_mode is None:
        rounding_mode = 0  # No rounding
    elif rounding_mode == 'trunc':
        rounding_mode = 1
    elif rounding_mode == 'floor':
        rounding_mode = 2
    else:
        raise ValueError("Invalid rounding_mode. Use None, 'trunc', or 'floor'.")

    # Prepare output tensor
    if out is None:
        out = tl.empty((num_tensors, n_elements), dtype=tl.float32)  # Adjust dtype as needed
    out_ptr = tl.make_tensor(out)

    # Launch the kernel
    grid = (num_tensors, n_elements)
    fused_hstack_div_kernel[grid](tensors_ptr, divisor_ptr, out_ptr, num_tensors, rounding_mode, n_elements)

    return out
