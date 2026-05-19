import triton
import triton.language as tl
import torch

@triton.jit
def _elementwise_op_kernel(
    x_ptr, y_ptr, output_ptr, size: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_size = 1024
    block_start = pid * block_size
    offsets = block_start + tl.arange(0, block_size)
    mask = offsets < size

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y  # Example element-wise operation: addition
    tl.store(output_ptr + offsets, output, mask=mask)

def broadcast_tensors(*tensors):
    # Use PyTorch's built-in broadcasting to handle the tensors
    broadcasted_tensors = torch.broadcast_tensors(*tensors)
    
    # Example: Perform an element-wise operation using the Triton kernel
    if len(broadcasted_tensors) == 2:
        x, y = broadcasted_tensors
        output = torch.empty_like(x)
        
        # Define the grid and block sizes
        grid = lambda meta: (x.numel() // meta['BLOCK_SIZE'] + 1,)
        _elementwise_op_kernel[grid](x, y, output, x.numel(), BLOCK_SIZE=1024)
        
        # Return the output tensor
        return [output, output]
    else:
        return broadcasted_tensors

# Example usage
x = torch.arange(3).view(1, 3)
y = torch.arange(2).view(2, 1)
a, b = broadcast_tensors(x, y)
print(a.size())  # Output: torch.Size([2, 3])
print(a)  # Output: tensor([[0, 1, 2], [0, 1, 2]])

x = torch.arange(3).view(1, 3)
y = torch.arange(2).view(2, 1)
a, b = broadcast_tensors(x, y)
assert a.size() == torch.Size([2, 3])
assert torch.all(a == torch.tensor([[0, 1, 2], [0, 1, 2]]))
