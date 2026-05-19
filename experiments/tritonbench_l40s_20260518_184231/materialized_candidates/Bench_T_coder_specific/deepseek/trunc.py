import triton
import triton.language as tl

@triton.jit
def trunc(x_ptr, y_ptr, N, **kwargs):
    # Define the grid
    tid = tl.program_id(axis=0)
    # Load x into shared memory
    x = tl.load(x_ptr + tid)
    # Compute the truncated value
    y = tl.trunc(x)
    # Store the result
    tl.store(y_ptr + tid, y)
