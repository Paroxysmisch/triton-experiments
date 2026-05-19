BLOCK_SIZE], dtype=tl.float32)
    cols = tl.arange(0, BLOCK_SIZE)
    input_ptrs = input_ptr + cols
    x = tl.load(input_ptrs, mask=cols < N, other=0.0).to(tl.float32)
    x_square = x * x
    mean = tl.sum(x_square, axis=0) / N
    var = tl.maximum(mean, eps)
    rstd = 1 / tl.sqrt(var)

    mask = cols < N
    w = tl.load(weights_ptr + cols, mask=mask)
    output = (x * w * rstd).to(DTYPE)
    output_ptrs = output_ptr + cols
    tl.store(output_ptrs, output, mask=mask)

@triton.jit
def fused_gather_masked_fill(
    input_ptr,
    dim,
    index_ptr,
    mask_ptr,
    value,
    output_ptr,
    input_strides,
    output_strides,
    input_sizes,
    index_sizes,
    mask_sizes,
    ndims,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Gather values from the input tensor along a specified dimension, and then
    replace the gathered elements with a specified value where the mask is True.

    Args:
        input_ptr (tensor): The input tensor.
        dim (int): The dimension along which to index.
        index_ptr (tensor): The indices of elements to gather.
        mask_ptr (tensor): A boolean mask tensor, broadcastable to the shape of the output tensor.
        value (float): The value to fill in where `mask` is True.
        output_ptr (tensor): Output tensor.
        input_strides (tensor): The strides for the input tensor.
        output_strides (tensor): The strides for the output tensor.
        input_sizes (tensor): The sizes for the input tensor.
        index_sizes (tensor): The sizes for the index tensor.
        mask_sizes (tensor): The sizes for the mask tensor.
        ndims (int): The number of dimensions in the input tensor.
        BLOCK_SIZE (constexpr): The block size for the kernel.
    """
    # Calculate the offsets for each block
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Load the index, mask, and output pointers
    index_ptrs = index_ptr + offsets
    mask_ptrs = mask_ptr + offsets
    output_ptrs = output_ptr + offsets

    # Load the index, mask, and output
    index = tl.load(index_ptrs, mask=tl.where(offsets < index_sizes[0], 1, 0), other=-1)
    mask = tl.load(mask_ptrs, mask=tl.where(offsets < mask_sizes[0], 1, 0), other=-1)

    # Calculate the input offsets
    input_offsets = tl.zeros((ndims,), dtype=tl.int32)
    input_offsets[dim] = index
    input_ptrs = input_ptr + tl.dot(input_offsets, input_strides)

    # Load the input
    input_ = tl.load(input_ptrs, mask=tl.where(offsets < input_sizes[0], 1, 0), other=-1)

    # Replace the input with the value where the mask is True
    output = tl.where(mask, value, input_)

    # Store the result
    tl.store(output_ptrs, output, mask=tl.where(offsets < output_sizes[0], 1, 0))
