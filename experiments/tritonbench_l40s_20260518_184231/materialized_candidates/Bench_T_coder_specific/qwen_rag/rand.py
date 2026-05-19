import triton
import triton.language as tl

@triton.jit
def _rand_triton(out_ptr: tl.tensor, n_elements: tl.int32):
    """
    Generates random float32 numbers in [0, 1) for each element in the output tensor.

    Args:
        out_ptr: The output tensor.
        n_elements: The number of elements in the output tensor.
    """
    pid = tl.program_id(axis=0)
    block_size = tl.block_dim(axis=0)
    grid_size = tl.cdiv(n_elements, block_size)

    offsets = pid * block_size + tl.arange(0, block_size)
    offsets = offsets[:n_elements]

    seed = pid
    out1, out2, out3, out4 = tl.rand4x(seed, offsets)

    tl.store(out_ptr + offsets, out1, mask=offsets < n_elements)
    if n_elements > block_size:
        offsets += block_size
        tl.store(out_ptr + offsets, out2, mask=offsets < n_elements)
    if n_elements > 2 * block_size:
        offsets += block_size
        tl.store(out_ptr + offsets, out3, mask=offsets < n_elements)
    if n_elements > 3 * block_size:
        offsets += block_size
        tl.store(out_ptr + offsets, out4, mask=offsets < n_elements)
