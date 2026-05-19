import triton
import triton.language as tl

# Define the Airy function kernel
@triton.jit
def airy_ai_kernel(
    input_ptr,
    output_ptr,
    N,
    BLOCK_SIZE: tl.constexpr
):
    # Get the global index
    idx = tl.program_id(axis=0)
    offset = idx * BLOCK_SIZE
    mask = offset + tl.arange(0, BLOCK_SIZE) < N

    # Load the input values
    x = tl.load(input_ptr + offset, mask=mask)

    # Constants for the Airy function
    C1 = 1 / (3**(2/3) * tl.math.pi)
    num_terms = 10  # Number of terms in the power series

    # Initialize the result
    result = tl.zeros_like(x, dtype=x.dtype)

    # Compute the Airy function using a power series approximation
    for k in range(num_terms):
        term = (-1)**k / (tl.math.factorial(k) * tl.math.gamma(k + 2/3))
        term *= (x / 3)**(3*k/2)
        result += term

    # Scale the result by C1
    result *= C1

    # Store the result
    tl.store(output_ptr + offset, result, mask=mask)
