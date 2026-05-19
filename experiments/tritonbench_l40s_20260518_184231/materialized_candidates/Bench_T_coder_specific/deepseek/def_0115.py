import triton
import triton.language as tl

@triton.jit
def tanh_linear_kernel(input_ptr, weight_ptr, output_ptr, bias_ptr,
                       in_features, out_features, batch_size):
    # Define the indices
    row_idx = tl.program_id(axis=0)
    col_idx = tl.program_id(axis=1)
    batch_idx = tl.program_id(axis=2)

    # Load the input and weight
    input_val = tl.load(input_ptr + batch_idx * in_features + row_idx, scope='shared')
    weight_val = tl.load(weight_ptr + col_idx * in_features + row_idx, scope='shared')
    bias_val = tl.load(bias_ptr + col_idx, scope='shared') if bias_ptr is not None else 0

    # Perform the linear transformation and activation
    output_val = tl.dot(input_val, weight_val) + bias_val
    output_val = tl.tanh(output_val)

    # Store the output
    tl.store(output_ptr + batch_idx * out_features + col_idx, output_val, scope='shared')

def tanh_linear(input, weight, bias=None):
    # Get the shapes
    batch_size, in_features = input.shape
    out_features, _ = weight.shape

    # Allocate the output
    output = triton.output(triton.float32, (batch_size, out_features))

    # Call the kernel
    tanh_linear_kernel[batch_size, out_features](input, weight, output, bias,
                                                in_features, out_features, batch_size)

    return output
