import triton
import triton.language as tl

@triton.jit
def bessel_i0_kernel(
        x_ptr,
        y_ptr,
        N,
        BLOCK_SIZE: tl.constexpr
):
    # Get the global index
    idx = tl.program_id(axis=0)
    offset = idx * BLOCK_SIZE
    mask = offset < N

    # Load input data with masking
    x = tl.load(x_ptr + offset, mask=mask, other=0.0)

    # Initialize variables for the sum
    y = tl.zeros_like(x)
    term = tl.ones_like(x)
    k = tl.zeros_like(x, dtype=tl.int32)

    while True:
        # Compute the next term in the series
        term *= (x * x / 4.0) / ((k * k + 1))
        y += term

        # Increment k
        k += 1

        # Check for convergence or maximum iterations
        if tl.any(tl.abs(term) <= 1e-6):
            break

    # Store the result with masking
    tl.store(y_ptr + offset, y, mask=mask)
