def adaptive_avg_pool2d(input, output_size):
    # Compute the output dimensions
    if type(output_size) is tuple:
        H, W = output_size
    else:
        H = W = output_size

    # Compute the spatial dimensions of the input
    N, C, H_in, W_in = input.shape

    # Compute the spatial dimensions of the output
    H_out = H if H is not None else H_in
    W_out = W if W is not None else W_in

    # Initialize the output tensor
    output = torch.zeros((N, C, H_out, W_out))

    # Compute the size of the receptive field
    h_step = H_in // H_out
    w_step = W_in // W_out

    # Perform the average pooling
    for n in range(N):
        for c in range(C):
            for h in range(H_out):
                for w in range(W_out):
                    h_start = h * h_step
                    h_end = min(h_start + h_step, H_in)
                    w_start = w * w_step
                    w_end = min(w_start + w_step, W_in)
                    output[n, c, h, w] = input[n, c, h_start:h_end, w_start:w_end].mean()

    return output
