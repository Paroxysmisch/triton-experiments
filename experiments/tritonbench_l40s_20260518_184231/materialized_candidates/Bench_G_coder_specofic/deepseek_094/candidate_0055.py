@triton.jit
def _swiglu_forward_kernel(a_ptr, b_ptr, c_ptr, n, BLOCK_SIZE, num_warps):
    # Define your SiLU activation function
    def silu(x):
        return 0.5 * x * (1.0 + tl.sigmoid(x))

    # Define your loop structure
    for i in range(n):
        # Load data
        a = tl.load(a_ptr + i * BLOCK_SIZE, num_warps)
        b = tl.load(b_ptr + i * BLOCK_SIZE, num_warps)

        # Apply SiLU and store the result
        c = silu(a) * b
        tl.store(c_ptr + i * BLOCK_SIZE, c, num_warps)
