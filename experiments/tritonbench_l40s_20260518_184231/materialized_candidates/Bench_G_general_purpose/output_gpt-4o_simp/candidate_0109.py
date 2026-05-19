import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel_online_v2(input_ptr, output_ptr, M, N, TILE_N, **meta):
    # Compute the row and column indices for this program
    row_idx = tl.program_id(0)
    col_idx = tl.arange(0, TILE_N)

    # Load the current row of the input matrix
    row_start = row_idx * N
    input_row = tl.load(input_ptr + row_start + col_idx, mask=col_idx < N, other=-float('inf'))

    # Compute the maximum value for numerical stability
    row_max = tl.max(input_row, axis=0)

    # Subtract the maximum and exponentiate
    exp_row = tl.exp(input_row - row_max)

    # Compute the sum of exponentials
    sum_exp = tl.sum(exp_row, axis=0)

    # Normalize the exponentials to get the softmax probabilities
    softmax_row = exp_row / sum_exp

    # Store the result in the output matrix
    tl.store(output_ptr + row_start + col_idx, softmax_row, mask=col_idx < N)

def prev_multiple_of(a, b):
    return (a // b) * b

def softmax(x):
    # Get the dimensions of the input tensor
    M, N = x.shape

    # Determine the tile size (must be a power of 2)
    TILE_N = 1 << (N - 1).bit_length()  # Closest power of 2 greater than or equal to N

    # Initialize the output tensor
    out = torch.empty_like(x)

    # Launch the Triton kernel
    grid = (M,)  # One program per row
    num_warps = 4  # You can tune this based on your hardware
    softmax_kernel_online_v2[grid](x, out, M, N, TILE_N, num_warps=num_warps)

    return out

# Example usage
x = torch.randn(128, 512, device='cuda')  # Example input tensor
out = softmax(x)
