import triton
import triton.language as tl

@triton.jit
def tensordot_rsqrt_kernel(
    a_ptr,
    b_ptr,
    output_ptr,
    a_shape,
    b_shape,
    strides_a,
    strides_b,
    num_elements,
    block_size: tl.constexpr,
):
    # Compute global indices
    idx = tl.program_id(0)
    coords_a = [idx // strides_a[i] % a_shape[i] for i in range(len(a_shape))]
    coords_b = [idx // strides_b[i] % b_shape[i] for i in range(len(b_shape))]

    # Initialize sum for the dot product
    dot_product = 0.0

    # Perform the dot product
    for k in range(num_elements):
        dot_product += a_ptr[coords_a + [k]] * b_ptr[coords_b + [k]]

    # Compute the reciprocal of the square root
    rsqrt_dot_product = 1 / tl.sqrt(dot_product)

    # Store the result
    output_ptr[idx] = rsqrt_dot_product
