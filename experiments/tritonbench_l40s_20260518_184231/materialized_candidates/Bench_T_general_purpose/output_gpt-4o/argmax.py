import triton
import triton.language as tl
import torch

@triton.jit
def argmax_kernel(input_ptr, output_ptr, stride, n_elements, dim, keepdim, BLOCK_SIZE: tl.constexpr):
    # Get the program ID
    pid = tl.program_id(axis=0)
    
    # Compute the offset for this block
    block_start = pid * BLOCK_SIZE
    
    # Create a range of indices for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Load input data
    data = tl.load(input_ptr + offsets * stride, mask=offsets < n_elements, other=-float('inf'))
    
    # Compute the argmax for this block
    max_idx = tl.argmax(data, axis=0)
    
    # Store the result
    tl.store(output_ptr + pid, max_idx)

def argmax(input, dim=None, keepdim=False):
    # If dim is None, flatten the input tensor
    if dim is None:
        input = input.flatten()
        dim = 0

    # Prepare the output tensor
    output_shape = list(input.shape)
    if not keepdim:
        output_shape.pop(dim)
    else:
        output_shape[dim] = 1
    output = torch.empty(output_shape, dtype=torch.long, device=input.device)

    # Calculate the stride for the dimension to reduce
    stride = input.stride(dim)
    n_elements = input.size(dim)

    # Launch the Triton kernel
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    argmax_kernel[grid](input, output, stride, n_elements, dim, keepdim, BLOCK_SIZE=BLOCK_SIZE)

    return output

# Example usage:
# input_tensor = torch.tensor([[1, 3, 2], [4, 6, 5]], device='cuda')
# result = argmax(input_tensor, dim=1, keepdim=True)
# print(result)
