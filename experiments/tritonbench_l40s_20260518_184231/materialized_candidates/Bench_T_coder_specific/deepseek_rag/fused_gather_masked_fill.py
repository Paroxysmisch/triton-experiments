def rms_norm(input, weight):
    # Get the device
    device = input.device
    # Get the dtype
    dtype = input.dtype
    # Get the shape
    shape = input.shape
    # Allocate output tensor
    output = torch.empty(shape, dtype=dtype, device=device)
    # Calculate stride
    stride = input.stride(0)
    # Get number of elements
    N = shape[-1]
    # Define epsilon
    eps = 1e-8
    # Define block size
    BLOCK_SIZE = 256
    # Define grid size
    grid = (shape[0],)
    # Define block size
    block = (BLOCK_SIZE,)
    # Create RMSNorm object
    rms_norm_obj = RMSNorm(input.data_ptr(), output.data_ptr(), weight.data_ptr(), stride, N, eps, dtype, BLOCK_SIZE)
    # Perform forward pass
    rms_norm_obj.forward()
    # Return output
    return output
