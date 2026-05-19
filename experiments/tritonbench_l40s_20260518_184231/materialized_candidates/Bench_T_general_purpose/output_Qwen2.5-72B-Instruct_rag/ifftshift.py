import triton
import triton.language as tl

@triton.jit
def ifftshift_kernel(input_ptr, output_ptr, shape, strides, num_dims, dims_to_shift, BLOCK_SIZE: tl.constexpr):
    """
    Triton kernel to perform ifftshift on the input tensor.
    
    Args:
        input_ptr: Pointer to the input tensor.
        output_ptr: Pointer to the output tensor.
        shape: Shape of the input tensor.
        strides: Strides of the input tensor.
        num_dims: Number of dimensions of the input tensor.
        dims_to_shift: Array of dimensions to rearrange.
        BLOCK_SIZE: Block size for parallel processing.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    # Compute the linear index in the input tensor
    linear_index = block_start + tl.arange(0, BLOCK_SIZE)
    in_bounds = tl.all(linear_index < shape[0] * shape[1] * shape[2] * shape[3])

    # Initialize output index to the same as input index
    output_index = linear_index

    if in_bounds:
        # Compute the multi-dimensional index from the linear index
        multi_index = [0] * num_dims
        for i in range(num_dims - 1, -1, -1):
            multi_index[i] = linear_index % shape[i]
            linear_index //= shape[i]

        # Apply ifftshift to the specified dimensions
        for dim in dims_to_shift:
            half_dim = shape[dim] // 2
            if multi_index[dim] < half_dim:
                multi_index[dim] += half_dim
            else:
                multi_index[dim] -= half_dim

        # Compute the new linear index for the output tensor
        new_linear_index = 0
        for i in range(num_dims):
            new_linear_index += multi_index[i] * strides[i]

        # Load input value and store it in the output tensor
        input_value = tl.load(input_ptr + linear_index)
        tl.store(output_ptr + new_linear_index, input_value)

import torch
import triton
import triton.language as tl

def ifftshift(input, dim=None):
    """
    Inverse of torch.fft.fftshift. Rearranges the elements of the input tensor
    such that the zero-frequency component is moved back to the original position.
    
    Args:
        input (Tensor): The tensor in FFT order.
        dim (int, Tuple[int], optional): The dimensions to rearrange.
            Only dimensions specified here will be rearranged, any other dimensions
            will be left in their original order.
            Default: All dimensions of input.
    
    Returns:
        Tensor: The rearranged tensor.
    """
    if dim is None:
        dim = tuple(range(input.dim()))
    elif isinstance(dim, int):
        dim = (dim,)
    
    # Convert input tensor to Triton tensor
    input_t = input.contiguous()
    output_t = torch.empty_like(input_t)
    
    # Prepare kernel launch configuration
    shape = input_t.shape
    strides = input_t.stride()
    num_dims = input_t.dim()
    dims_to_shift = torch.tensor(dim, dtype=torch.int32, device=input_t.device)
    
    # Launch the kernel
    grid = (triton.cdiv(input_t.numel(), 1024),)
    ifftshift_kernel[grid](input_t, output_t, shape, strides, num_dims, dims_to_shift, BLOCK_SIZE=1024)
    
    return output_t

# Sample tensor
input_tensor = torch.arange(16).reshape(4, 4)

# Apply ifftshift using the wrapper function
output_tensor = ifftshift(input_tensor, dim=(0, 1))

# Expected result using torch.fft.ifftshift
expected_tensor = torch.fft.ifftshift(input_tensor, dim=(0, 1))

# Verify the results
print("Output Tensor:\n", output_tensor)
print("Expected Tensor:\n", expected_tensor)
assert torch.allclose(output_tensor, expected_tensor), "The output does not match the expected result."
