import triton
import triton.language as tl

@triton.jit
def ifftshift_kernel(
    input_ptr,
    output_ptr,
    strides,
    input_shape,
    output_shape,
    num_elements,
    block_size,
    grid_size,
    dim=None
):
    pid = tl.program_id(axis=0)
    tid = tl.program_id(axis=1)
    x = pid * block_size + tid

    if x >= num_elements:
        return

    coords = []
    for i in range(len(input_shape)):
        coord = x // strides[i] % input_shape[i]
        if dim is None or i in dim:
            shift = input_shape[i] // 2
            if coord < shift:
                coord += input_shape[i]
        coords.append(coord)

    output_coords = tuple(coords)
    output_idx = sum(output_coords[i] * output_strides[i] for i in range(len(output_shape)))
    tl.store(output_ptr + output_idx, tl.load(input_ptr + x))

@triton.jit
def ifftshift(input, dim=None):
    """
    Rearranges the elements of the input tensor such that the zero-frequency component is moved back to the original position.

    Args:
        input (Tensor): the tensor in FFT order
        dim (int, Tuple[int], optional): The dimensions to rearrange.
            Only dimensions specified here will be rearranged, any other dimensions
            will be left in their original order.
            Default: All dimensions of input.

    Returns:
        Tensor: the tensor after ifftshift operation
    """
    input_shape = input.shape
    output_shape = input_shape
    num_elements = np.prod(input_shape)
    strides = np.cumprod((1,) + input_shape[::-1])[::-1]
    output_strides = np.cumprod((1,) + output_shape[::-1])[::-1]

    # Allocate output tensor
    output = tl.zeros_like(input)

    # Launch kernel
    block_size = 256
    grid_size = (num_elements + block_size - 1) // block_size
    ifftshift_kernel[input_shape, block_size](input.data, output.data, strides, input_shape, output_shape, num_elements, block_size, grid_size, dim)

    return output
