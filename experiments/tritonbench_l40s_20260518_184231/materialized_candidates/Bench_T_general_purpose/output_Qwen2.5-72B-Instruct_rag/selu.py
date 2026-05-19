import triton
import triton.language as tl
import torch

# Constants for SELU
alpha = 1.6732632423543772848170429916717
scale = 1.0507009873554804934193349852946

# Triton kernel for SELU
@triton.jit
def _selu_kernel(X, Y, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(X + offsets, mask=mask)
    
    # Compute SELU
    x_pos = tl.where(x > 0, x, 0)
    x_neg = tl.where(x <= 0, alpha * (tl.exp(x) - 1), 0)
    y = scale * (x_pos + x_neg)
    
    # Store the result
    tl.store(Y + offsets, y, mask=mask)

# Wrapper function for SELU
def selu(input, inplace=False):
    assert input.is_cuda, "Input tensor must be on a CUDA device"
    
    # Determine the output tensor
    if inplace:
        output = input
    else:
        output = torch.empty_like(input)
    
    # Launch the kernel
    n_elements = input.numel()
    grid = (triton.cdiv(n_elements, 1024),)
    _selu_kernel[grid](input, output, n_elements, BLOCK_SIZE=1024)
    
    return output

# Example usage
if __name__ == "__main__":
    # Create a random input tensor
    x = torch.randn(1024, device='cuda')
    
    # Apply SELU
    y = selu(x, inplace=False)
    print(y)
    
    # Apply SELU in place
    selu(x, inplace=True)
    print(x)
