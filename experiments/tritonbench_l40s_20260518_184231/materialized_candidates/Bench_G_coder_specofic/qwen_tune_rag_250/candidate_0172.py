softmax(input):
    # Softmax is just a scaled exponential
    # So we can use the trick from PyTorch's implementation
    # to make it more numerically stable
    MAX_NUM = 20

    # We only want to reshape if necessary, since it creates a new tensor.
    reshaped = input.unsqueeze(0) if input.ndim == 1 else input
    reshaped = reshaped.flatten(0, -2)

    # Triton kernel requires power-of-2
    next_power_of_2 = 2 ** (reshaped.shape[1] - 1).bit_length()

    # The kernel will read from and write to the same memory location,
    # so we allocate a temporary buffer if necessary.
    output = torch.empty_like(reshaped)
    if reshaped.stride(-1) != 1:
        output = torch.empty(reshaped.shape, dtype=reshaped.dtype, device=reshaped.device)
    else:
        output.copy_(reshaped)

    # Launch kernel
    grid = lambda meta: (reshaped.numel() // meta['BLOCK_SIZE'], )

    softmax_kernel[grid](reshaped, output, reshaped.stride(0), output.stride(0), reshaped.shape[1], BLOCK_SIZE=next_power_of_2)

    # If we allocated a temporary buffer, copy the result back.
    if output is not reshaped:
        output.copy_(reshaped)

    return output.view_as(input)
