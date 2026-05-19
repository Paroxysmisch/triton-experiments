import triton
import triton.language as tl

@triton.jit
def rmsnorm_triton(x_ptr, rms_w_ptr, out_ptr, N_SIZE, eps, BLOCK_N_SIZE, batch_stride, M_stride, K_stride, num_warps):
    # Obtain the program ids for batch and M dimensions
    batch_id = tl.program_id(0)
    M_id = tl.program_id(1)

    # Calculate the base pointers for the current block
    x_block_ptr = x_ptr + batch_id * batch_stride + M_id * M_stride
    out_block_ptr = out_ptr + batch_id * batch_stride + M_id * M_stride

    # Initialize a variable to accumulate the sum of squares
    sum_of_squares = tl.zeros((BLOCK_N_SIZE,), dtype=tl.float32)

    # Iterate over the K dimension in chunks of BLOCK_N_SIZE
    for k_offset in range(0, K_stride, BLOCK_N_SIZE):
        # Load a block of data from x
        x_block = tl.load(x_block_ptr + k_offset, mask=True)

        # Accumulate the sum of squares
        sum_of_squares += x_block * x_block

    # Compute the RMS value
    rms_value = tl.sqrt(sum_of_squares / N_SIZE + eps)

    # Normalize and scale the input
    for k_offset in range(0, K_stride, BLOCK_N_SIZE):
        # Load a block of data from x
        x_block = tl.load(x_block_ptr + k_offset, mask=True)

        # Normalize and scale
        normalized_block = (x_block / rms_value) * tl.load(rms_w_ptr + k_offset, mask=True)

        # Store the result
        tl.store(out_block_ptr + k_offset, normalized_block, mask=True)

def rmsnorm_wrapper(x, rms_weights, eps, BLOCK_N_SIZE, num_warps):
    # Extract dimensions
    batch, M, K = x.shape

    # Allocate output tensor
    out = torch.empty_like(x)

    # Calculate strides
    batch_stride = M * K
    M_stride = K
    K_stride = K

    # Launch the Triton kernel
    grid = (batch, M)
    rmsnorm_triton[grid](x, rms_weights, out, K, eps, BLOCK_N_SIZE, batch_stride, M_stride, K_stride, num_warps)

    return out
