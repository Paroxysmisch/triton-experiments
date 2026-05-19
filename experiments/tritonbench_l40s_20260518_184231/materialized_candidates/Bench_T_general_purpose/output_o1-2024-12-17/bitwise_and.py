import triton
import triton.language as tl


@triton.jit
def _bitwise_and_kernel(
    input_ptr, other_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask=mask, other=0)
    y = tl.load(other_ptr + offsets, mask=mask, other=0)
    z = x & y

    tl.store(out_ptr + offsets, z, mask=mask)


def bitwise_and(input, other, *, out=None):
    # Ensure the input, other, and out have the same number of elements or can be broadcast
    # For simplicity, assume they are the same shape in this example.
    assert input.dtype in ('bool', 'int8', 'uint8', 'int16', 'uint16',
                           'int32', 'uint32', 'int64', 'uint64'), "input must be integral or boolean"
    assert other.dtype in ('bool', 'int8', 'uint8', 'int16', 'uint16',
                           'int32', 'uint32', 'int64', 'uint64'), "other must be integral or boolean"

    if out is None:
        out = input.clone()  # or allocate a new tensor with the same shape/dtype

    n_elements = input.numel()
    BLOCK_SIZE = 1024
    grid = ( (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE, )

    triton.run(
        _bitwise_and_kernel,
        grid=grid,
        num_warps=4,
        args=[
            input, other, out,
            n_elements
        ],
        kwargs={
            'BLOCK_SIZE': BLOCK_SIZE
        }
    )

    return out
