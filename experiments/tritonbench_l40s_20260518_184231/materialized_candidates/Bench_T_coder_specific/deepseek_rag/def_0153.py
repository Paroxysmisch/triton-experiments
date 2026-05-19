@triton.jit
def adaptive_avg_pool2d(input, output, output_size):
    # Calculate the spatial dimensions of the output
    H_out, W_out = output_size
    H_in, W_in = input.shape[-2:]
    H_scale, W_scale = H_in / H_out, W_in / W_out

    # Iterate over the output pixels
    for i in range(H_out):
        for j in range(W_out):
            # Calculate the range of input pixels that contribute to the output pixel
            start_i, end_i = int(i * H_scale), int((i + 1) * H_scale)
            start_j, end_j = int(j * W_scale), int((j + 1) * W_scale)

            # Sum the values of the input pixels that contribute to the output pixel
            sum_value = 0
            for k in range(start_i, end_i):
                for l in range(start_j, end_j):
                    sum_value += input[k, l]

            # Calculate the average and store it in the output tensor
            output[i, j] = sum_value / ((end_i - start_i) * (end_j - start_j))
