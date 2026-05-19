import triton
import triton.language as tl
import torch
from typing import Optional

@triton.jit
def leaky_relu_kernel(
    output_ptr: tl.tensor,
    input_ptr: tl.tensor,
    negative_slope: float,
    BLOCK_SIZE: tl.constexpr):
    
    pid = tl.program_id(axis=0)
    grid_size = tl.cdiv(output_ptr.shape[0], BLOCK_SIZE)
    num_blocks = min(grid_size, 1024)
    tid = tl.arange(0, BLOCK_SIZE)
    i = pid * BLOCK_SIZE + tid
    
    # Check if out-of-bounds
    mask = i < output_ptr.shape[0]
    
    output_ptr[i] = tl.where(mask, 
                              tl.maximum(0.0, input_ptr[i]) + negative_slope * tl.minimum(0.0, input_ptr[i]), 
                              0.0)

class LeakyReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, negative_slope=0.01):
        output = input.new_zeros_like(input)
        n_elements = input.numel()
        block_size = 256
        grid_size = (n_elements + block_size - 1) // block_size
        
        leaky_relu_kernel[grid_size, block_size](output, input, negative_slope, n_elements)
        
        ctx.save_for_backward(input, negative_slope)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input, negative_slope = ctx.saved_tensors
        grad_input = grad_output.clone()
        
        n_elements = input.numel()
        block_size = 256
        grid_size = (n_elements + block_size - 1) // block_size
        
        grad_input[input < 0] *= negative_slope
        
        return grad_input, None

# Register the custom autograd function
leaky_relu = LeakyReLU.apply

# Example usage
input_tensor = torch.randn(1024, requires_grad=True)
output_tensor = leaky_relu(input_tensor)
loss = output_tensor.sum()
loss.backward()

print("Input:", input_tensor)
print("Output:", output_tensor)
print("Gradient:", input_tensor.grad)
