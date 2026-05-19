import torch
import triton
import triton.language as tl

@triton.jit
def sigmoid(x):
    # Apply the sigmoid function using the formula provided
    return 1 / (1 + tl.exp(-x))

@triton.jit
def sigmoid_argmax_kernel(input, output, n_elements,
                          stride_input_row, stride_output_row,
                          BLOCK_SIZE: tl.constexpr):
    # Compute the program ID
    pid = tl.program_id(0)
    # Calculate the start index for this program
    start = pid * BLOCK_SIZE
    # Create offsets for the block
    offsets = start + tl.arange(0, BLOCK_SIZE)
    # Create a mask for valid indices
    mask = offsets < n_elements
    # Load input elements with the mask
    x = tl.load(input + offsets, mask=mask)
    # Apply the sigmoid function
    y = sigmoid(x)
    # Store the result
    tl.store(output + offsets, y, mask=mask)

def sigmoid_argmax(input, dim=None, keepdim=False):
    # Ensure input is a tensor
    if not isinstance(input, torch.Tensor):
        input = torch.tensor(input)
    # If dim is None, compute the argmax over all elements
    if dim is None:
        result = torch.argmax(input)
        return result
    # Otherwise, reduce the dimension
    else:
        return torch.argmax(input, dim=dim, keepdim=keepdim)
