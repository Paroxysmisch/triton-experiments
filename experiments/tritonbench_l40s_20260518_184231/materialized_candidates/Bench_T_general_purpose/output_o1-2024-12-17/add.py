@triton.jit
def _add_kernel(
    input_ptr, other_ptr, alpha_ptr, out_ptr,
    input_offset, other_offset, out_offset,
    input_stride, other_stride, out_stride,
    input_broadcast, other_broadcast,
    n_elements: tl.constexpr
):
    pid = tl.program_id(0)
    block_size = tl.num_programs(0)
    offsets = pid * block_size + tl.arange(0, block_size)
    mask = offsets < n_elements

    # Load alpha as scalar
    alpha_val = tl.load(alpha_ptr, mask=[True])

    # Load input/other values, handle broadcasting
    input_val = tl.load(input_ptr + (offsets * input_stride * input_broadcast + input_offset), mask=mask, other=0)
    other_val = tl.load(other_ptr + (offsets * other_stride * other_broadcast + other_offset), mask=mask, other=0)

    # Perform the addition
    result = input_val + alpha_val * other_val

    # Store result
    tl.store(out_ptr + (offsets * out_stride + out_offset), result, mask=mask)


def add(input, other, *, alpha=1, out=None):
    """
    add(input, other, *, alpha=1, out=None) -> Tensor
    """
    import triton
    import triton.language as tl
    import math

    # Handle type promotion
    import numpy as np
    if not hasattr(input, "dtype"):
        input = np.array(input)
    if not hasattr(other, "dtype"):
        other = np.array(other)
    dtype = np.result_type(input, other, alpha)
    alpha = np.array(alpha, dtype=dtype)
    
    # Convert inputs to array if necessary
    input_arr = np.array(input, dtype=dtype)
    other_arr = np.array(other, dtype=dtype)

    # Determine broadcast shape
    broadcast_shape = np.broadcast(input_arr, other_arr).shape

    # Prepare output
    if out is None:
        out = np.zeros(broadcast_shape, dtype=dtype)
    else:
        if not hasattr(out, "dtype"):
            out = np.array(out)
        if out.shape != broadcast_shape:
            raise ValueError("Provided out has incorrect shape.")
        if out.dtype != dtype:
            raise ValueError("Provided out has incorrect dtype.")

    # Flatten arrays for contiguous memory
    input_arr_flat = input_arr.astype(dtype).ravel()
    other_arr_flat = other_arr.astype(dtype).ravel()
    out_flat = out.ravel()
    alpha_arr = alpha.ravel()

    # Allocate on device
    d_input = triton.language.device_put(input_arr_flat)
    d_other = triton.language.device_put(other_arr_flat)
    d_out = triton.language.device_put(out_flat)
    d_alpha = triton.language.device_put(alpha_arr)

    # Stride and broadcast:
    # For simplicity, assume contiguous data and compute broadcast manually
    n_input = input_arr_flat.size
    n_other = other_arr_flat.size
    # Since data is flattened, if shapes are different, we decide broadcast factor
    input_broadcast = 0 if n_input == 1 else 1
    other_broadcast = 0 if n_other == 1 else 1

    # Offsets are zero in flattened memory
    input_offset = 0
    other_offset = 0
    out_offset = 0

    # Stride is 1 for contiguous data
    input_stride = 1
    other_stride = 1
    out_stride = 1

    # Launch kernel
    n_elements = out_flat.size
    block_size = 1024
    grid = lambda meta: (math.ceil(n_elements / block_size),)
    
    _add_kernel[grid](
        d_input, d_other, d_alpha, d_out,
        input_offset, other_offset, out_offset,
        input_stride, other_stride, out_stride,
        input_broadcast, other_broadcast,
        n_elements,
        num_warps=4,
        num_stages=2,
        BLOCK_SIZE=block_size
    )

    # Copy result back
    out[...] = d_out.to_numpy().reshape(out.shape)
    return out
