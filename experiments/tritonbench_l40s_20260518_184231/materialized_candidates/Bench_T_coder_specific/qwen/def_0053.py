import triton
import triton.language as tl

@triton.jit
def mul_relu_kernel(
    X_ptr, Y_ptr, Out_ptr,
    N_elements: tl.int32,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Load data from global memory
    x = tl.load(X_ptr + offsets)
    y = tl.load(Y_ptr + offsets)

    # Element-wise multiplication
    z = x * y

    # Apply ReLU
    out = tl.maximum(z, 0.0)

    # Store results back to global memory
    tl.store(Out_ptr + offsets, out)
