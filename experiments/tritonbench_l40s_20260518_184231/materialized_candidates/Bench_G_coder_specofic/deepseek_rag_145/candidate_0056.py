import torch
import triton
import triton.language as tl

@triton.jit
def _add_kernel(A, B, C, SIZE, BLOCK: tl.constexpr):
    # determine the current index in the grid
    ptx = tl.program_id(0)
    # calculate the start index of the current block of elements
    start = ptx * BLOCK
    # calculate the end index of the current block of elements
    end = start + BLOCK
    # load elements from `a` and `b` into registers if they are within the range
    a = tl.load(A + start) if start < SIZE else 0
    b = tl.load(B + start) if start < SIZE else 0
    # calculate the sum and store it in `c` if it is within the range
    if start < SIZE:
        tl.store(C + start, a + b)

def custom_add(a, b):
    # create an empty tensor `c` with the same size as `a` and `b`
    c = torch.zeros_like(a)
    # calculate the size of `a` and `b`
    size = a.numel()
    # calculate the number of elements each program instance processes
    BLOCK = 16
    # calculate the number of program instances
    NBLOCKS = triton.cdiv(size, BLOCK)
    # launch the kernel
    _add_kernel[NBLOCKS, BLOCK](a, b, c, size, BLOCK=BLOCK)
    return c
