import triton
import triton.language as tl
import torch
import math

# Define the Triton kernel
@triton.jit
def reciprocal_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Calculate the global index
    pid = tl.program_id(axis=0)
    num_pid = tl.cdiv(n_elements, BLOCK_SIZE)

    # Calculate the local index within the block
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Mask to ensure we don't access out-of-bounds memory
    mask = offsets < n_elements

    # Load the data
    a_val = tl.load(a_ptr + offsets, mask=mask)

    # Compute the reciprocal
    b_val = 1.0 / a_val.to(tl.float32)

    # Store the result
    tl.store(b_ptr + offsets, b_val, mask=mask)

# Define the Python wrapper function
def reciprocal(input_tensor, out=None):
    # Determine the shape and dtype of the input tensor
    n_elements = input_tensor.numel()

    # Create an empty output tensor if one is not provided
    if out is None:
        out = torch.empty_like(input_tensor)

    # Determine the block size
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(n_elements)))

    # Determine the grid size
    grid_size = triton.cdiv(n_elements, block_size)

    # Launch the Triton kernel
    reciprocal_kernel[(grid_size, 1, 1)](input_tensor.data_ptr(), out.data_ptr(), n_elements, block_size)

    return out

# Example usage
if __name__ == "__main__":
    a = torch.tensor([1, 2, 3, 4], dtype=torch.int32, device="cuda")
    print("Input:", a)
    result = reciprocal(a)
    print("Reciprocal:", result)
