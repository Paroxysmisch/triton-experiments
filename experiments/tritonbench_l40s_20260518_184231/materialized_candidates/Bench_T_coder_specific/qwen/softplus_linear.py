import triton
import triton.language as tl

@triton.jit
def softplus_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    input_shape, weight_shape, bias_shape,
    stride_input_x, stride_input_y, stride_weight, stride_bias,
    beta, threshold,
    BLOCK_SIZE_X: tl.constexpr, BLOCK_SIZE_Y: tl.constexpr
):
    # Get global index
    x = tl.program_id(axis=0)
    y = tl.program_id(axis=1)

    # Calculate indices
    input_idx = y * stride_input_y + x
    weight_idx = 0
    bias_idx = 0

    # Load input and weight
    input_val = tl.load(input_ptr + input_idx)
    weight_val = tl.load(weight_ptr + weight_idx)

    # Perform linear transformation
    linear_result = input_val * weight_val

    # Add bias if provided
    if bias_shape[0] > 0:
        bias_val = tl.load(bias_ptr + bias_idx)
        linear_result += bias_val

    # Apply Softplus
    result = (1 / beta) * tl.log(1 + tl.exp(beta * linear_result))

    # Handle threshold for numerical stability
    mask = linear_result > threshold
    result[mask] = linear_result[mask]

    # Store result
    tl.store(output_ptr + input_idx, result)
