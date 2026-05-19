import triton
import triton.language as tl
import torch

# Kernel to multiply elements by 2 and store in a new tensor
@triton.jit
def mul2_kernel(X, Z, n_elements, BLOCK_SIZE: tl.constexpr):
    # Compute offsets for the current block
    offsets = tl.arange(0, BLOCK_SIZE)
    # Load data from the input tensor
    x = tl.load(X + offsets, mask=offsets < n_elements, other=0.0)
    # Multiply by 2
    z = x * 2
    # Store the result in the output tensor
    tl.store(Z + offsets, z, mask=offsets < n_elements)

# Kernel to multiply elements by 2 in-place
@triton.jit
def mul2_inplace_kernel(X, n_elements, BLOCK_SIZE: tl.constexpr):
    # Compute offsets for the current block
    offsets = tl.arange(0, BLOCK_SIZE)
    # Load data from the input tensor
    x = tl.load(X + offsets, mask=offsets < n_elements, other=0.0)
    # Multiply by 2
    x *= 2
    # Store the result back to the input tensor
    tl.store(X + offsets, x, mask=offsets < n_elements)

# Wrapper function for mul2_kernel
def triton_mul2(input_tensor):
    n_elements = input_tensor.numel()
    output_tensor = torch.empty_like(input_tensor)
    # Launch the kernel
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    mul2_kernel[grid](input_tensor, output_tensor, n_elements, BLOCK_SIZE=1024)
    return output_tensor

# Wrapper function for mul2_inplace_kernel
def triton_mul2_inplace(input_tensor):
    n_elements = input_tensor.numel()
    # Launch the kernel
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    mul2_inplace_kernel[grid](input_tensor, n_elements, BLOCK_SIZE=1024)

# Example usage
input_tensor = torch.randn(1024, device='cuda')
output_tensor = triton_mul2(input_tensor)
triton_mul2_inplace(input_tensor)
