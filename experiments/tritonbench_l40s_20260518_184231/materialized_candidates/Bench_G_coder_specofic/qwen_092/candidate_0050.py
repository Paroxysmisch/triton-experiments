import triton
import triton.language as tl

@triton.jit
def relu_kernel(
    x_ptr: tl.tensor,  # Pointer to input tensor
    y_ptr: tl.tensor,  # Pointer to output tensor
    N: tl.int32,        # Size of the input tensor
    BLOCK_SIZE: tl.constexpr  # Block size for parallel processing
):
    # Each program will process one element of the input tensor
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < N  # Mask to ensure we don't access out-of-bounds elements

    # Load input values
    x = tl.load(x_ptr + indices, mask=mask)

    # Apply ReLU activation
    y = tl.where(x >= 0, x, 0)

    # Store results back to output tensor
    tl.store(y_ptr + indices, y, mask=mask)
