import torch
import triton
import triton.language as tl

# Triton kernel
@triton.jit
def fused_add_mul_activation_kernel(
    x_ptr, bias_ptr, in_ptr, out_ptr,
    BLOCK_SIZE: tl.constexpr, multiplier: tl.constexpr, activation_type: tl.constexpr
):
    # Define block start position
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    # Create a block of indices
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Load data from memory
    x = tl.load(x_ptr + offsets, mask=offsets < x_ptr.shape[0], other=0.0)
    bias = tl.load(bias_ptr + offsets, mask=offsets < bias_ptr.shape[0], other=0.0)
    in_val = tl.load(in_ptr + offsets, mask=offsets < in_ptr.shape[0], other=0.0)

    # Perform the fused operation: (x + bias + multiplier * in_val)
    result = x + bias + multiplier * in_val

    # Apply activation function
    if activation_type == 'sigmoid':
        result = 1 / (1 + tl.exp(-result))
    elif activation_type == 'relu':
        result = tl.max(result, 0)

    # Store the result
    tl.store(out_ptr + offsets, result, mask=offsets < out_ptr.shape[0])

# PyTorch wrapper
def fused_add_mul_activation_torch(in_out_tensor, bias, in_tensor, multiplier=1.0, activation_type='relu'):
    # Ensure the input tensors are on the same device
    assert in_out_tensor.device == bias.device == in_tensor.device

    # Define the block size
    BLOCK_SIZE = 1024

    # Calculate the grid size
    grid_size = (in_out_tensor.numel() + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Allocate output tensor
    out_tensor = torch.empty_like(in_out_tensor)

    # Launch the Triton kernel
    fused_add_mul_activation_kernel[grid_size](
        in_out_tensor, bias, in_tensor, out_tensor,
        BLOCK_SIZE=BLOCK_SIZE, multiplier=multiplier, activation_type=activation_type
    )

    return out_tensor

# Example usage
if __name__ == "__main__":
    # Initialize tensors
    in_out_tensor = torch.randn(10240, device='cuda')
    bias = torch.randn(10240, device='cuda')
    in_tensor = torch.randn(10240, device='cuda')

    # Call the wrapper function
    result = fused_add_mul_activation_torch(in_out_tensor, bias, in_tensor, multiplier=0.5, activation_type='sigmoid')

    # Print the result
    print(result)
