import triton
import triton.language as tl

@triton.jit
def bessel_j1_kernel(
        x_ptr,
        y_ptr,
        N,
        BLOCK_SIZE: tl.constexpr
):
    # Get the global index for the current program
    idx = tl.program_id(axis=0)
    offset = idx * BLOCK_SIZE

    # Get the slice of the input tensor for this program
    x = tl.load(x_ptr + offset, mask=(offset + BLOCK_SIZE <= N), other=0.0)

    # Approximation of Bessel function of the first kind of order 1
    # Using the series expansion for small x: j1(x) ~ x / 2
    y = x / 2.0

    # Store the result in the output tensor
    tl.store(y_ptr + offset, y, mask=(offset + BLOCK_SIZE <= N))
