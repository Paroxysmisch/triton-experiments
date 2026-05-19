import torch
import triton
import triton.language as tl

# Define the Triton kernel
@triton.jit
def mul_kernel(src_ptr, dst_ptr, BLOCK_SIZE: tl.constexpr):
    # Define the exponent compensator
    exponent_compensator = 2.0 ** (127 - 15)

    # Get the block index
    block_idx = tl.program_id(0)

    # Compute the start index for this block
    start_idx = block_idx * BLOCK_SIZE

    # Create an index range for this block
    offsets = tl.arange(0, BLOCK_SIZE)

    # Compute the actual indices in the source tensor
    indices = start_idx + offsets

    # Load elements from the source tensor
    src = tl.load(src_ptr + indices)

    # Perform the multiplication
    result = src * exponent_compensator

    # Store the result in the destination tensor
    tl.store(dst_ptr + indices, result)

# Define the wrapper function
def launch_mul_kernel(src, BLOCK_SIZE=1024):
    # Ensure src is a CUDA tensor
    assert src.is_cuda, "Input tensor must be on CUDA device"

    # Calculate the number of blocks needed
    num_blocks = (src.numel() + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Create an empty destination tensor on the same device as src
    dst = torch.empty_like(src)

    # Launch the Triton kernel
    mul_kernel[(num_blocks,)](src, dst, BLOCK_SIZE=BLOCK_SIZE)

    # Return the destination tensor
    return dst

# Example usage
if __name__ == "__main__":
    # Create a source tensor on CUDA
    src = torch.rand(4096, device='cuda')

    # Launch the kernel
    dst = launch_mul_kernel(src, BLOCK_SIZE=1024)

    # Print the result
    print(dst)
