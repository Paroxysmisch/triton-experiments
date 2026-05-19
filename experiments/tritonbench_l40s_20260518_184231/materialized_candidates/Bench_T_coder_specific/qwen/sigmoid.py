import torch
import triton
import triton.language as tl

# Define the Triton kernel
sigmoid_kernel = triton.compile(sigmoid_kernel, signature="f32[:], f32[:], i32")

def sigmoid(input, *, out=None):
    if out is None:
        out = torch.empty_like(input)

    # Check if the input and output tensors have the same shape
    assert input.shape == out.shape, "Input and output tensors must have the same shape"

    # Launch the Triton kernel
    grid_size = (input.numel() + 255) // 256
    sigmoid_kernel[(grid_size,), (256,)](input.contiguous().data_ptr(), out.data_ptr(), input.numel())

    return out
