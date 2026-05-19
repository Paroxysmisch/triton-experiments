def ones_like_triton(input, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format):
    # Define the output tensor with the same size as the input tensor
    output = torch.ones_like(input, dtype=dtype, layout=layout, device=device, requires_grad=requires_grad, memory_format=memory_format)
    return output
