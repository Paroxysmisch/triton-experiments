import triton
import triton.language as tl

@triton.jit
def sigmoid_kernel(x_ptr,  # *Pointer* to first input vector.
                   output_ptr,  # *Pointer* to output vector.
                   n_elements,  # Size of the vector.
                   BLOCK_SIZE: tl.constexpr):  # Number of elements each program should process.

    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    e_x = tl.exp(-x)
    output = 1 / (1 + e_x)

    tl.store(output_ptr + offsets, output, mask=mask)
