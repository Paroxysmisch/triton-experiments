import triton
import triton.language as tl

@triton.jit
def ifftshift_kernel(
    input_ptr,  # Pointer to the input tensor
    output_ptr,  # Pointer to the output tensor
    input_stride,  # Stride of the input tensor
    output_stride,  # Stride of the output tensor
    input_shape,  # Shape of the input tensor
    dim,  # Dimensions to rearrange
    BLOCK_SIZE: tl.constexpr,  # Block size for parallelization
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    # Iterate over the elements in the block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < input_shape

    # Load the input elements
    input_elements = tl.load(input_ptr + offsets * input_stride, mask=mask)

    # Calculate the new indices for ifftshift
    new_indices = tl.where(offsets < input_shape // 2, offsets + input_shape // 2, offsets - input_shape // 2)

    # Store the elements in the output tensor
    tl.store(output_ptr + new_indices * output_stride, input_elements, mask=mask)

import torch
import triton
import triton.language as tl

def ifftshift(input, dim=None):
    if dim is None:
        dim = list(range(input.dim()))
    elif isinstance(dim, int):
        dim = [dim]
    
    output = torch.empty_like(input)

    for d in dim:
        input_shape = input.shape[d]
        input_stride = input.stride(d)
        output_stride = output.stride(d)

        grid = (input.numel() // input_shape, )
        ifftshift_kernel[grid](
            input_ptr=input.flatten().contiguous().data_ptr(),
            output_ptr=output.flatten().contiguous().data_ptr(),
            input_stride=input_stride,
            output_stride=output_stride,
            input_shape=input_shape,
            dim=d,
            BLOCK_SIZE=1024,
        )

    return output
