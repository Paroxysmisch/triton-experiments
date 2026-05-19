import triton
import triton.language as tl

@triton.jit
def add_mean_kernel(
    input_ptr,
    other_ptr,
    output_ptr,
    alpha,
    stride_input,
    stride_other,
    stride_output,
    n_elements,
    n_dims,
    dims,
    keepdim,
    dtype
):
    pid = tl.program_id(axis=0)
    grid_size = tl.cdiv(n_elements, stride_input)

    # Load input and other values
    input_val = tl.load(input_ptr + pid * stride_input, mask=pid < n_elements, other=0.0)
    other_val = tl.load(other_ptr + pid * stride_other, mask=pid < n_elements, other=0.0)

    # Scale other by alpha
    scaled_other = alpha * other_val

    # Add scaled other to input
    result = input_val + scaled_other

    # Reduce over dimensions
    if n_dims > 0:
        for d in range(n_dims):
            dim = dims[d]
            reduction_idx = pid % stride_input // stride_input ** dim
            reduction_mask = reduction_idx == 0
            result = tl.where(reduction_mask, result, 0.0)
            result = tl.sum(result, axis=dims[d], keepdim=True)
    
    # Store result
    tl.store(output_ptr + pid * stride_output, result, mask=pid < n_elements)
