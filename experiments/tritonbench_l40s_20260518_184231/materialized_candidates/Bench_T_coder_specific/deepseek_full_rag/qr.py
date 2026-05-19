ptr.stride(0), N, eps)
    """
    # Calculate offsets of elements accessed by this program instance
    offsets = tl.arange(0, BLOCK_SIZE)
    # Load input data; pad out-of-bounds indices with 0
    x = tl.load(input_ptr + offsets, mask=offsets < N, other=0)
    tl.debug_barrier()
    # Compute variance
    x_square = x * x
    x_square_mean = tl.sum(x_square) / N
    # Add epsilon to avoid division by zero
    rms = tl.sqrt(x_square_mean + eps)
    # Normalize
    normalized_x = x / rms
    # Scale and shift
    y = normalized_x * weights_ptr
    # Write output
    tl.store(output_ptr + offsets, y, mask=offsets < N)
