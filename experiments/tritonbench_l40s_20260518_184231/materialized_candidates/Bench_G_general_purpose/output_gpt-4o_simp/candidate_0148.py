import torch
import triton
import triton.language as tl

# Define the block size
BLOCK_SIZE = 1024

# Kernel to multiply elements by 2 and store in a new tensor
@triton.jit
def mul2_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Program ID
    pid = tl.program_id(axis=0)

    # Create block of indices
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Load input data with mask to prevent out-of-bounds access
    mask = offsets < n_elements
    input_data = tl.load(input_ptr + offsets, mask=mask, other=0.0)

    # Multiply by 2
    result = input_data * 2.0

    # Store result in output tensor
    tl.store(output_ptr + offsets, result, mask=mask)

# Kernel to multiply elements by 2 in place
@triton.jit
def mul2_inplace_kernel(input_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Program ID
    pid = tl.program_id(axis=0)

    # Create block of indices
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Load input data with mask to prevent out-of-bounds access
    mask = offsets < n_elements
    input_data = tl.load(input_ptr + offsets, mask=mask, other=0.0)

    # Multiply by 2
    result = input_data * 2.0

    # Store result back in input tensor
    tl.store(input_ptr + offsets, result, mask=mask)

def triton_mul2(input_tensor):
    # Ensure input is a contiguous tensor
    input_tensor = input_tensor.contiguous()
    n_elements = input_tensor.numel()

    # Create an output tensor
    output_tensor = torch.empty_like(input_tensor)

    # Launch kernel
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    mul2_kernel[grid](input_tensor, output_tensor, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    return output_tensor

def triton_mul2_inplace(input_tensor):
    # Ensure input is a contiguous tensor
    input_tensor = input_tensor.contiguous()
    n_elements = input_tensor.numel()

    # Launch kernel
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    mul2_inplace_kernel[grid](input_tensor, n_elements, BLOCK_SIZE=BLOCK_SIZE)

# Example usage
if __name__ == "__main__":
    # Create a sample tensor
    x = torch.tensor([1.0, 2.0, 3.0, 4.0], device='cuda')

    # Double the values using triton_mul2
    y = triton_mul2(x)
    print("Output (new tensor):", y)

    # Double the values in place using triton_mul2_inplace
    triton_mul2_inplace(x)
    print("Output (in place):", x)
