import triton
import triton.language as tl

@triton.jit
def chunk_global_reversed_cumsum_scalar_kernel(
    s: tl.tensor,  # Input tensor of shape (B, H, T)
    o: tl.tensor,  # Output tensor of shape (B, H, T)
    B: tl.int32,     # Batch size
    H: tl.int32,     # Number of heads
    T: tl.int32,     # Sequence length
    BT: tl.int32,    # Block size along T dimension
):
    # Calculate the global index
    b_idx = tl.program_id(0) // H
    h_idx = tl.program_id(0) % H
    b = b_idx
    h = h_idx

    # Initialize the accumulation variable
    b_z = 0.0

    # Iterate over the dimension T in blocks of size BT
    for t in range(T - 1, -1, -1):
        # Load the current block
        block_start = max(t - BT + 1, 0)
        block_end = t + 1
        block_size = block_end - block_start
        block_sum = 0.0

        for i in range(block_size):
            block_sum += s[b * H * T + h * T + block_start + i]

        # Update the accumulation variable
        b_z += block_sum

        # Calculate the cumulative sum by subtracting it from the sum
        cumsum = block_sum - b_z

        # Store the result in the output tensor
        o[b * H * T + h * T + t] = cumsum

def chunk_global_reversed_cumsum_scalar(s, o, B, H, T, BT):
    # Check input shapes
    assert s.shape == (B, H, T)
    assert o.shape == (B, H, T)

    # Launch the kernel
    grid = (B * H,)
    block = (1,)

    # Call the kernel
    chunk_global_reversed_cumsum_scalar_kernel[grid, block](s, o, B, H, T, BT)

# Example usage
import torch

# Example tensors
B, H, T = 2, 3, 10
BT = 4

s = torch.randn(B, H, T, device='cuda')
o = torch.zeros_like(s)

# Convert to Triton tensors
s_triton = triton.from_dlpack(s.detach().cpu().numpy())
o_triton = triton.from_dlpack(o.detach().cpu().numpy())

# Call the wrapper function
chunk_global_reversed_cumsum_scalar(s_triton, o_triton, B, H, T, BT)

# Convert back to PyTorch tensors
o = triton.to_dlpack(o_triton).to('cuda')

print(o)
