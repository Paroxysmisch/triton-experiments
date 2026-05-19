import triton
import triton.language as tl

BLOCK_SIZE = 1024

@triton.jit
def _dropout(x_ptr, x_keep_ptr, output_ptr, n_elements, p):
    # get the block id and element id
    block_id = tl.program_id(0)
    element_id = tl.arange(0, BLOCK_SIZE)

    # calculate the offsets
    offset = block_id * BLOCK_SIZE + element_id

    # check if the offset is within bounds
    mask = offset < n_elements

    # load the elements
    x = tl.load(x_ptr + offset, mask=mask)
    x_keep = tl.load(x_keep_ptr + offset, mask=mask)

    # perform the operation
    y = tl.where(x_keep, x / (1 - p), 0.0)

    # store the result
    tl.store(output_ptr + offset, y, mask=mask)

def dropout(x, x_keep, output, p):
    # ensure input tensor `x` is contiguous
    x = triton.testing.make_contiguous(x)

    # compute the grid size
    n_elements = x.numel()
    n_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    # call the Triton kernel
    _dropout[n_blocks, BLOCK_SIZE](x, x_keep, output, n_elements, p)
