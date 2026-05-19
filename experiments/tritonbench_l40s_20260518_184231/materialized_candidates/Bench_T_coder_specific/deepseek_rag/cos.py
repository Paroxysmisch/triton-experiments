import torch
import triton
import triton.language as tl

@triton.jit
def cos_kernel(input_ptr, output_ptr, n):
    # Get the index of the current program
    idx = tl.program_id(0)
    # Calculate the offsets for the current program
    offsets = idx * tl.constexpr(n)
    # Load the input values from input_ptr
    x = tl.load(input_ptr + offsets)
    # Compute the cosine of the input values
    y = tl.cos(x)
    # Store the result in output_ptr
    tl.store(output_ptr + offsets, y)

def cos(input: torch.Tensor, out=None):
    # Get the number of elements in the input tensor
    n = input.numel()
    # If an output tensor is not provided, create a new tensor
    if out is None:
        output = torch.empty(size=input.shape).to(input.device)
    else:
        output = out
    # Ensure both input and output tensors are on the same device
    assert input.device == output.device, 'Input and output tensors must be on the same device'
    # Launch the Triton kernel
    cos_kernel[(n+127)//128, 128](input_ptr=input, output_ptr=output, n=n)
    return output
