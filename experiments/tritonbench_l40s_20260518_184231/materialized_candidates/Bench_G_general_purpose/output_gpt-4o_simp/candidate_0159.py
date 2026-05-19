import triton
import triton.language as tl
import torch

@triton.jit
def nested3_kernel(in_ptr, out_ptr, stride_n, num_tiles, BLOCK_SIZE: tl.constexpr):
    # Block indices
    block_x = tl.program_id(0)
    block_y = tl.program_id(1)

    # Calculate base pointers for this block
    base_x = block_x * BLOCK_SIZE
    base_y = block_y * BLOCK_SIZE

    # Iterate over 2x2 tiles within the block
    for i in range(2):
        for j in range(2):
            # Calculate the offsets for the current tile
            offset_x = base_x + i
            offset_y = base_y + j

            # Calculate the linear index
            idx = offset_y * stride_n + offset_x

            # Load from input
            data = tl.load(in_ptr + idx)

            # Store to output
            tl.store(out_ptr + idx, data)

def wrapper_nested3(input_tensor, stride_n):
    # Ensure the input tensor is on the GPU
    input_tensor = input_tensor.to('cuda')
    
    # Create an output tensor of the same shape
    output_tensor = torch.empty_like(input_tensor)

    # Define the grid size
    num_tiles_x = input_tensor.shape[1] // 2
    num_tiles_y = input_tensor.shape[0] // 2
    grid = (num_tiles_x, num_tiles_y)

    # Launch the kernel
    nested3_kernel[grid](
        input_tensor,
        output_tensor,
        stride_n,
        num_tiles_x * num_tiles_y,
        BLOCK_SIZE=2
    )

    # Print the output tensor
    print(output_tensor.cpu())

# Example usage
if __name__ == "__main__":
    # Define the input tensor and stride
    input_tensor = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    stride_n = input_tensor.stride(0)

    # Call the wrapper function
    wrapper_nested3(input_tensor, stride_n)
