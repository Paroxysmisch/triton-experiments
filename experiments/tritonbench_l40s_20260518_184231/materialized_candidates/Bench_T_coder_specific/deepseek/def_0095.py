import triton as tl

def batch_norm(input, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-05):
    # Define the dimensions
    N, C, *spatial = input.shape
    input = tl.reshape(input, [N, C, -1])

    # Apply Batch Normalization
    output = tl.batch_normalization(input, running_mean, running_var, weight, bias, training, momentum, eps)

    # Reshape the output back to its original shape
    output = tl.reshape(output, [N, C, *spatial])

    return output
