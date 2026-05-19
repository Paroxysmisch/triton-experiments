@triton.jit
def adaptive_avg_pool2d_kernel(input, output, N, C, H_in, W_in, H_out, W_out):
    # Calculate the stride for pooling
    stride_h = H_in // H_out
    stride_w = W_in // W_out

    # Iterate over the output dimensions
    for n in range(N):
        for c in range(C):
            for h in range(H_out):
                for w in range(W_out):
                    # Calculate the pooling region
                    h_start = h * stride_h
                    w_start = w * stride_w
                    h_end = min(h_start + stride_h, H_in)
                    w_end = min(w_start + stride_w, W_in)

                    # Compute the average
                    sum_val = 0.0
                    count = 0
                    for i in range(h_start, h_end):
                        for j in range(w_start, w_end):
                            sum_val += input[n, c, i, j]
                            count += 1
                    output[n, c, h, w] = sum_val / count

def adaptive_avg_pool2d(output_size) -> Tensor:
    # Validate output_size
    if isinstance(output_size, int):
        H_out, W_out = output_size, output_size
    elif isinstance(output_size, tuple) and len(output_size) == 2:
        H_out, W_out = output_size
    else:
        raise ValueError("output_size must be an int or a tuple of two ints.")

    # Get input tensor shape
    input_tensor = ...  # Assume this is provided
    N, C, H_in, W_in = input_tensor.shape

    # Handle None values for output size
    if H_out is None:
        H_out = H_in
    if W_out is None:
        W_out = W_in

    # Prepare output tensor
    output_tensor = torch.empty((N, C, H_out, W_out), dtype=input_tensor.dtype)

    # Launch the Triton kernel
    adaptive_avg_pool2d_kernel[(N, C, H_out, W_out)](input_tensor, output_tensor, N, C, H_in, W_in, H_out, W_out)

    return output_tensor
