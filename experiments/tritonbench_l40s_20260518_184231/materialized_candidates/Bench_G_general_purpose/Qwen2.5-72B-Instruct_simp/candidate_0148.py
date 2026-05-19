import triton
import triton.language as tl
import torch

# Define the block size
BLOCK_SIZE = 1024

# Kernel to double the values in the input tensor and store the result in a new tensor
@triton.jit
def mul2_kernel(
    x_ptr,  # Pointer to the input tensor
    y_ptr,  # Pointer to the output tensor
    n_elements,  # Number of elements in the tensor
    BLOCK_SIZE: tl.constexpr,  # Block size
):
    # Compute the block ID
    pid = tl.program_id(axis=0)
    # Compute the start and end indices for this block
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask to avoid out-of-bounds memory access
    mask = offsets < n_elements
    # Load the data from the input tensor
    x = tl.load(x_ptr + offsets, mask=mask)
    # Perform the multiplication
    y = x * 2
    # Store the result in the output tensor
    tl.store(y_ptr + offsets, y, mask=mask)

# Kernel to double the values in the input tensor in place
@triton.jit
def mul2_inplace_kernel(
    x_ptr,  # Pointer to the input tensor
    n_elements,  # Number of elements in the tensor
    BLOCK_SIZE: tl.constexpr,  # Block size
):
    # Compute the block ID
    pid = tl.program_id(axis=0)
    # Compute the start and end indices for this block
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask to avoid out-of-bounds memory access
    mask = offsets < n_elements
    # Load the data from the input tensor
    x = tl.load(x_ptr + offsets, mask=mask)
    # Perform the multiplication
    y = x * 2
    # Store the result back in the input tensor
    tl.store(x_ptr + offsets, y, mask=mask)

# Wrapper function to call the mul2_kernel
def triton_mul2(x: torch.Tensor) -> torch.Tensor:
    # Ensure the input tensor is on the GPU
    x = x.cuda()
    # Create an output tensor with the same shape and type as the input tensor
    y = torch.empty_like(x)
    # Compute the number of elements in the tensor
    n_elements = x.numel()
    # Compute the number of blocks needed
    grid = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    # Launch the kernel
    mul2_kernel[grid, BLOCK_SIZE](x, y, n_elements, BLOCK_SIZE)
    # Return the output tensor
    return y

# Wrapper function to call the mul2_inplace_kernel
def triton_mul2_inplace(x: torch.Tensor) -> None:
    # Ensure the input tensor is on the GPU
    x = x.cuda()
    # Compute the number of elements in the tensor
    n_elements = x.numel()
    # Compute the number of blocks needed
    grid = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    # Launch the kernel
    mul2_inplace_kernel[grid, BLOCK_SIZE](x, n_elements, BLOCK_SIZE)

# Example usage
if __name__ == "__main__":
    # Create a PyTorch tensor
    x = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)
    print("Original tensor:", x)

    # Double the values and store the result in a new tensor
    y = triton_mul2(x)
    print("Doubled tensor (new tensor):", y)

    # Double the values in place
    triton_mul2_inplace(x)
    print("Doubled tensor (in place):", x)
