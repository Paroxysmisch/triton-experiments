import triton
import triton.language as tl

@triton.jit
def abs_kernel(
        A_ptr,
        O_ptr,
        N,
        BLOCK_SIZE: tl.constexpr
):
    # Get the global index of the current thread
    idx = tl.program_id(axis=0)
    # Calculate the offset for the current element
    offset = idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Create a mask to handle out-of-bounds accesses
    mask = offset < N

    # Load the input element with masking
    a = tl.load(A_ptr + offset, mask=mask)
    # Compute the absolute value
    abs_a = tl.abs(a)
    # Store the result in the output tensor with masking
    tl.store(O_ptr + offset, abs_a, mask=mask)
