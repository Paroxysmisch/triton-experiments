@triton.jit
def _fused_index_select_eq(
    input_ptr,
    input_stride,
    index_ptr,
    index_stride,
    other_ptr,
    other_stride,
    output_ptr,
    output_stride,
    N,
    as_scalar,
    BLOCK_SIZE: tl.constexpr,
):
    # Define the grid
    grid = tl.grid(BLOCK_SIZE, BLOCK_SIZE)

    # Perform the index selection and element-wise equality comparison
    for i in range(N):
        for j in range(N):
            # Compute the input index
            input_idx = input_stride * i
            # Compute the index for the output
            output_idx = output_stride * i
            # Perform the index selection
            x = tl.load(input_ptr + input_idx)
            # Perform the element-wise equality comparison
            y = tl.load(other_ptr + other_stride * j)
            if as_scalar:
                y = y[0]
            z = (x == y)
            tl.store(output_ptr + output_idx, z)

def fused_index_select_eq(input, dim, index, other, out=None):
    # Get the shapes of the input and output
    input_shape = list(input.shape)
    if out is None:
        output_shape = list(input.shape)
        output_shape[dim] = index.numel()
        out = torch.empty(output_shape, device=input.device, dtype=torch.bool)
    else:
        output_shape = list(out.shape)

    # Compute the strides for the input, output, and index tensors
    input_stride = tl.arange(len(input_shape))
    output_stride = tl.arange(len(output_shape))
    index_stride = tl.arange(len(index.shape))

    # Compute the pointers to the input, output, and index tensors
    input_ptr = input.data_ptr()
    output_ptr = out.data_ptr()
    index_ptr = index.data_ptr()

    # Compute the pointers to the other tensor or scalar
    other_ptr = other.data_ptr() if isinstance(other, torch.Tensor) else other
    other_stride = 1 if isinstance(other, torch.Tensor) else 0

    # Call the Triton kernel
    _fused_index_select_eq[grid](
        input_ptr,
        input_stride[dim],
        index_ptr,
