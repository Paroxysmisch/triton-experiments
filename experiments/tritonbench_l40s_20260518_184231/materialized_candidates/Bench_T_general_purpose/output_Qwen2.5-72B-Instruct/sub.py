import triton
import triton.language as tl

@triton.jit
def sub_kernel(
    input_ptr,  # Pointer to the input tensor
    other_ptr,  # Pointer to the other tensor or number
    output_ptr, # Pointer to the output tensor
    input_n_elements,  # Number of elements in the input tensor
    other_n_elements,  # Number of elements in the other tensor
    alpha,  # Scalar multiplier for the other tensor
    input_stride,  # Stride of the input tensor
    other_stride,  # Stride of the other tensor
    output_stride,  # Stride of the output tensor
    BLOCK_SIZE: tl.constexpr  # Block size for parallel processing
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < input_n_elements

    input_offsets = offsets * input_stride
    other_offsets = (offsets % other_n_elements) * other_stride
    output_offsets = offsets * output_stride

    input_vals = tl.load(input_ptr + input_offsets, mask=mask)
    other_vals = tl.load(other_ptr + other_offsets, mask=mask)

    output_vals = input_vals - alpha * other_vals
    tl.store(output_ptr + output_offsets, output_vals, mask=mask)

import torch
import triton
import triton.language as tl

def sub(input, other, *, alpha=1, out=None):
    # Ensure input and other are tensors
    if not isinstance(input, torch.Tensor):
        input = torch.tensor(input)
    if not isinstance(other, torch.Tensor):
        other = torch.tensor(other)

    # Ensure input and other are broadcastable
    input, other = torch.broadcast_tensors(input, other)

    # Ensure the output tensor is provided or created
    if out is None:
        out = torch.empty_like(input)

    # Ensure the tensors are on the same device
    assert input.device == other.device == out.device, "All tensors must be on the same device"

    # Get the number of elements and strides
    input_n_elements = input.numel()
    other_n_elements = other.numel()
    input_stride = input.stride(0)
    other_stride = other.stride(0)
    output_stride = out.stride(0)

    # Define the grid and block sizes
    grid = (triton.cdiv(input_n_elements, 1024),)
    block_size = 1024

    # Launch the Triton kernel
    sub_kernel[grid](
        input.data_ptr(),
        other.data_ptr(),
        out.data_ptr(),
        input_n_elements,
        other_n_elements,
        alpha,
        input_stride,
        other_stride,
        output_stride,
        BLOCK_SIZE=block_size
    )

    return out
