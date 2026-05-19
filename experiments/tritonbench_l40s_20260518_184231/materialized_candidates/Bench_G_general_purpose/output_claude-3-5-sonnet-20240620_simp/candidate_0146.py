import torch
import triton
import triton.language as tl

@triton.jit
def fused_add_mul_activation_kernel(
    # Pointers to tensors
    x_ptr,     # Pointer to first input tensor
    y_ptr,     # Pointer to second input tensor
    z_ptr,     # Pointer to third input tensor (output)
    n_elements,  # Size of the tensors
    BLOCK_SIZE: tl.constexpr,  # Number of elements per block
):
    # Calculate the absolute position of the block
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    # Create offsets for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create a mask to handle the case where the tensor size
    # is not a multiple of BLOCK_SIZE
    mask = offsets < n_elements
    
    # Load data from x and y
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    
    # Perform fused operations:
    # 1. Add x and y
    # 2. Multiply by 2 (as an example multiplier)
    # 3. Apply ReLU activation
    output = x + y
    output = output * 2.0
    output = tl.maximum(output, 0.0)  # ReLU activation
    
    # Store the result
    tl.store(z_ptr + offsets, output, mask=mask)

# PyTorch wrapper function
def fused_add_mul_activation_torch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # Input validation
    assert x.shape == y.shape, "Input tensor shapes must match"
    assert x.is_cuda and y.is_cuda, "Input tensors must be on GPU"
    assert x.dtype == y.dtype, "Input tensors must have the same dtype"
    
    # Output tensor
    output = torch.empty_like(x)
    
    # Calculate launch grid
    n_elements = output.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    fused_add_mul_activation_kernel[grid](
        x.data_ptr(),
        y.data_ptr(),
        output.data_ptr(),
        n_elements,
        BLOCK_SIZE,
    )
    
    return output

# Example usage
x = torch.randn(1000, device='cuda')
y = torch.randn(1000, device='cuda')
result = fused_add_mul_activation_torch(x, y)
