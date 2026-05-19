import triton
import triton.language as tl
import numpy as np

# Define the block sizes
BLOCK_M = 128
BLOCK_N = 128
BLOCK_K = 64

# Define the sliding window size
WINDOW_SIZE = 4

# Define the function to execute the kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': BLOCK_M, 'BLOCK_N': BLOCK_N, 'BLOCK_K': BLOCK_K}, num_stages=1, num_warps=4),
    ],
    key=['Q', 'K', 'M']
)
def get_score(Q, K, M, Out, sm_scale, window_size, sliding_window):
    # Determine the grid size
    grid_size = (Q.shape[0] // BLOCK_M, K.shape[0] // BLOCK_N, Q.shape[2] // BLOCK_K)

    # Execute the kernel
    _score_kernel[grid_size](Q, K, M, Out, BLOCK_M, BLOCK_N, BLOCK_K, sm_scale, window_size, sliding_window)

# Define the function to calculate the scale factor for attention
def calculate_sm_scale(dim):
    return 1.0 / np.sqrt(dim)

# Define the main function to run the attention score computation
def attention_score(Q, K, M, sliding_window=False):
    # Determine the output shape
    out_shape = (Q.shape[0], K.shape[0], Q.shape[2])

    # Allocate memory for the output
    Out = np.zeros(out_shape, dtype=np.float32)

    # Calculate the scale factor for attention
    sm_scale = calculate_sm_scale(Q.shape[1])

    # Execute the kernel
    get_score(Q, K, M, Out, sm_scale, WINDOW_SIZE, sliding_window)

    return Out
