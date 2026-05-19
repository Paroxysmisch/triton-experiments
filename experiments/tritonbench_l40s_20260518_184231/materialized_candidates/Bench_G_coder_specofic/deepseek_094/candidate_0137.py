import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements):
    # get the index of the current thread
    pid = tl.program_id(axis=0)
    # calculate the base index for each block
    block_start = pid * tl.program_id(axis=1)
    # calculate the offsets within the block
    offsets = block_start + tl.arange(0, n_elements)
    # apply a mask to prevent out-of-bound memory access
    mask = offsets < n_elements
    # load the values from the input pointers
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    # perform the addition
    out = x + y
    # store the result back in the output pointer
    tl.store(out_ptr + offsets, out, mask=mask)

def add_wrapper(x, y):
    # initialize the output tensor
    out = torch.zeros_like(x)
    # calculate the total number of elements
    n_elements = x.numel()
    # calculate the number of blocks needed
    num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    # launch the kernel
    add_kernel[(num_blocks,)](x, y, out, n_elements)
    return out
