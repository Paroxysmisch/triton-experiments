import triton
import triton.language as tl

# Define the maximum number of terms in the series expansion
MAX_TERMS = 10

@triton.jit
def erf_kernel(X_ptr, Y_ptr, N, BLOCK_SIZE: tl.constexpr):
    # Get the global index within the grid
    pid = tl.program_id(axis=0)
    # Calculate the starting index for this thread block
    start_idx = pid * BLOCK_SIZE
    # Get the range of indices to process
    offsets = start_idx + tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't access out-of-bound indices
    mask = offsets < N

    # Load the input data
    x = tl.load(X_ptr + offsets, mask=mask)

    # Initialize the sum for the series expansion
    sum_series = tl.zeros_like(x)
    factorial = tl.tensor(1.0, dtype=x.dtype)
    power = tl.tensor(1.0, dtype=x.dtype)

    # Perform the series expansion
    for k in range(MAX_TERMS):
        term = power / factorial
        sum_series += term
        power *= -x * x
        factorial *= (2 * k + 3) * (2 * k + 4)

    # Normalize and store the result
    result = 2.0 / tl.sqrt(tl.f32(math.pi)) * sum_series
    tl.store(Y_ptr + offsets, result, mask=mask)
