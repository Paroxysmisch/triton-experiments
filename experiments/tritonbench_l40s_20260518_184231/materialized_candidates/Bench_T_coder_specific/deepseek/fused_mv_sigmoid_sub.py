@triton.jit
def _kernel(input_ptr, vec_ptr, other_ptr, out_ptr,
            N, M, alpha,
            BLOCK_SIZE: tl.constexpr):
    # Get the thread id
    pid = tl.program_id(axis=0)
    # Compute the sigmoid function
    def sigmoid(x):
        return 1 / (1 + tl.exp(-x))
    # Loop over the blocks
    for block_start in range(0, N, BLOCK_SIZE):
        # Compute the block end
        block_end = min(block_start + BLOCK_SIZE, N)
        # Loop over the rows
        for i in range(block_start + pid, block_end, BLOCK_SIZE):
            # Compute the matrix-vector product
            z = tl.dot(input_ptr[i * M:(i + 1) * M], vec_ptr)
            # Apply the sigmoid function
            s = sigmoid(z)
            # Subtract alpha * other from s
            y = s - alpha * other_ptr[i]
            # Store the result
            out_ptr[i] = y
