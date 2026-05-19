import triton
import triton.language as tl

# Define the Triton kernel
@triton.jit
def chunk_global_cumsum_vector_kernel(
    s_ptr, z_ptr, stride, N, BT, BS, BLOCK: tl.constexpr
):
    # Program ID for each block
    pid = tl.program_id(axis=0)
    # Block ID within the grid
    bid = pid // (N // BLOCK)
    # Remaining blocks to process
    remaining_blocks = pid % (N // BLOCK)
    # Block index within the batch
    b = bid // (N // BLOCK)
    # Block index within the head
    h = bid % (N // BLOCK)
    # Time index within the block
    t = remaining_blocks // (N // BLOCK)
    # Index within the block
    i = remaining_blocks % (N // BLOCK)
    
    # Lower triangular mask for the block
    m_s = tl.make_mask(tl.arange(0, BS), i)
    
    # Pointer to the relevant data slice in s
    s_ptr = s_ptr + (b * N + h * N + t * N + i) * stride
    z_ptr = z_ptr + (b * N + h * N + t * N + i) * stride
    
    # Load data from s into b_s and convert to float32
    b_s = tl.load(s_ptr, mask=m_s, other=0.0).to(tl.float32)
    
    # Initialize block-level cumulative sum
    b_c = b_s
    
    # Perform block-level cumulative sum using matrix multiplication
    for j in range(BS):
        b_c = b_c + b_s * m_s
    
    # Store the result back to the output tensor z
    tl.store(z_ptr, b_c, mask=m_s)
    
    # Update the running sum
    b_z = b_c
    for j in range(BS):
        b_z = b_z + b_s * m_s

# Define the Python wrapper function
def chunk_global_cumsum_vector(s, BT, BS):
    # Get the tensor dimensions
    B, H, T, S = s.shape
    # Compute the grid size
    grid_size = (B * H * (T + BS - 1) // BS, 1)
    # Create an output tensor of the same shape
    z = tl.zeros_like(s)
    # Call the Triton kernel with the appropriate stride
    stride = B * H * T * S
    chunk_global_cumsum_vector_kernel[grid_size](s, z, stride, S, BT, BS, BLOCK=BS)
    return z
