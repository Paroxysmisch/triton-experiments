import triton
import triton.language as tl

# Define the Triton kernel
@triton.jit
def conv2d_forward_kernel(
    input_ptr, weight_ptr, output_ptr,
    input_shape, weight_shape, output_shape,
    input_strides, weight_strides, output_strides,
    kernel_size, stride, padding, groups,
    precision,
    BLOCK_SIZE_BATCH=32,
    BLOCK_SIZE_INPUT_FEATURES=16,
    BLOCK_SIZE_OUTPUT_FEATURES=16,
):
    """
    Triton kernel for 2D convolution.
    """
    batch, input_feature, output_feature, h, w = tl.program_id(0), tl.program_id(1), tl.program_id(2), tl.program_id(3), tl.program_id(4)
    
    # Compute the indices for input, weight, and output
    ih = h * stride - padding[0] + tl.arange(0, kernel_size[0])
    iw = w * stride - padding[1] + tl.arange(0, kernel_size[1])
    
    # Clamp the indices to ensure they are within the valid range
    ih = tl.clip(ih, 0, input_shape[2])
    iw = tl.clip(iw, 0, input_shape[3])
    
    # Load input and weight values
    input_val = tl.load(input_ptr + input_strides[0] * batch + input_strides[1] * input_feature + input_strides[2] * ih + input_strides[3] * iw)
    weight_val = tl.load(weight_ptr + weight_strides[0] * output_feature + weight_strides[1] * (input_feature // groups) + weight_strides[2] * ih + weight_strides[3] * iw)
    
    # Accumulate the result
    acc = tl.zeros([kernel_size[0], kernel_size[1]], dtype=input_val.dtype)
    acc += input_val * weight_val
    
    # Store the result in the output tensor
    tl.store(output_ptr + output_strides[0] * batch + output_strides[1] * output_feature + output_strides[2] * h + output_strides[3] * w, acc.sum())

# Define the Python wrapper function
def conv2d_forward(input_tensor, weight_tensor, kernel_size, stride, padding, groups=1, precision='fp32'):
    """
    Wrapper function for 2D convolution.
    """
    # Get input and weight shapes
    input_shape = input_tensor.shape
    weight_shape = weight_tensor.shape
    
    # Calculate output shape
    output_shape = (
        input_shape[0],  # batch size
        input_shape[1] // groups,  # output features
        (input_shape[2] + 2 * padding[0] - kernel_size[0]) // stride[0] + 1,  # output height
        (input_shape[3] + 2 * padding[1] - kernel_size[1]) // stride[1] + 1  # output width
    )
    
    # Calculate strides
    input_strides = [input_shape[1] * input_shape[2] * input_shape[3], input_shape[2] * input_shape[3], input_shape[3], 1]
    weight_strides = [output_shape[1] * output_shape[2] * output_shape[3], output_shape[2] * output_shape[3], output_shape[3], 1]
    output_strides = [output_shape[1] * output_shape[2] * output_shape[3], output_shape[2] * output_shape[3], output_shape[3], 1]
    
    # Allocate output tensor
    output_tensor = tl.zeros(output_shape, dtype=input_tensor.dtype)
    
    # Compute block and grid sizes
    block_size = (BLOCK_SIZE_BATCH, BLOCK_SIZE_INPUT_FEATURES, BLOCK_SIZE_OUTPUT_FEATURES)
    grid_size = (
        (output_shape[0] + block_size[0] - 1) // block_size[0],
        (output_shape[1] + block_size[1] - 1) // block_size[1],
        (output_shape[2] + block_size[2] - 1) // block_size[2]
    )
    
    # Launch the kernel
    conv2d_forward_kernel[grid_size, block_size](
        input_tensor.data_ptr(), weight_tensor.data_ptr(), output_tensor.data_ptr(),
        input_shape, weight_shape, output_shape,
        input_strides, weight_strides, output_strides,
        kernel_size, stride, padding, groups,
        precision
    )
    
    return output_tensor
