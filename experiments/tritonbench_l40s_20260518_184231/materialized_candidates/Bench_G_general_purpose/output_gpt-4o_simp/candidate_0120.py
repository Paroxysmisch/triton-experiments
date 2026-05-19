import torch
import triton
import triton.language as tl

# Kernel definition
@triton.jit
def chunk_global_cumsum_scalar_kernel(s_ptr, o_ptr, n_elements, BT, BLOCK_SIZE: tl.constexpr):
    # Compute the block index and the starting index for this block
    block_idx = tl.program_id(0)
    start_idx = block_idx * BLOCK_SIZE

    # Initialize a running total for the cumulative sum
    running_total = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    # Iterate over the elements in chunks
    for i in range(0, n_elements, BT):
        # Load a chunk of data from the input tensor
        offset = start_idx + i
        data = tl.load(s_ptr + offset, mask=offset < n_elements, other=0.0)

        # Compute the cumulative sum for this chunk
        running_total += data

        # Store the result in the output tensor
        tl.store(o_ptr + offset, running_total, mask=offset < n_elements)

# Wrapper function
def chunk_global_cumsum_scalar(s, BT):
    # Get the shape of the input tensor
    B, H, W = s.shape

    # Flatten the first two dimensions to simplify the kernel launch
    s_flat = s.view(-1, W)
    n_elements = s_flat.shape[1]

    # Prepare an output tensor of the same shape
    o_flat = torch.empty_like(s_flat)

    # Define the block size and grid size
    BLOCK_SIZE = 1024  # You can adjust this value based on your hardware
    grid_size = (s_flat.shape[0],)

    # Launch the kernel
    chunk_global_cumsum_scalar_kernel[grid_size](
        s_flat, o_flat, n_elements, BT, BLOCK_SIZE=BLOCK_SIZE
    )

    # Reshape the output tensor to the original shape
    o = o_flat.view(B, H, W)
    return o

# Example usage
if __name__ == "__main__":
    # Create a 3D tensor
    B, H, W = 4, 3, 8  # Example dimensions
    s = torch.rand((B, H, W), dtype=torch.float32).cuda()

    # Define the chunk size
    BT = 4

    # Perform chunked global cumsum
    o = chunk_global_cumsum_scalar(s, BT)

    # Print the result
    print(o)
