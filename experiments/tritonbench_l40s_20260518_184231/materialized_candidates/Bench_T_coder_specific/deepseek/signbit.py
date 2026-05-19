def signbit(input, out=None):
    if out is None:
        out = torch.empty_like(input)
    else:
        assert out.is_cuda and out.is_floating_point() and out.shape == input.shape, \
            "The output tensor must be a CUDA floating-point tensor with the same shape as the input tensor."

    N = input.numel()
    grid = (N + 255) // 256
    signbit_kernel[grid, 256](input.contiguous().view(-1), out.contiguous().view(-1), N)
    return out
