import triton.language as tl

@triton.jit
def sigmoid_argmax_kernel(input_ptr, output_ptr, n,
                          BLOCK_SIZE: tl.constexpr):
    # Define the sigmoid function
    def sigmoid(x):
        return 1 / (1 + tl.exp(-x))

    # Define the grid
    grid = tl.grid(n, BLOCK_SIZE)

    # Iterate over the grid
    for i in range(grid.n):
        # Compute the sigmoid of the input
        x = tl.load(input_ptr + i)
        y = sigmoid(x)

        # Store the result
        tl.store(output_ptr + i, y)
