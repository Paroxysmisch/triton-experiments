import triton
import triton.language as tl

@triton.jit
def cos_func(
    a_ptr,
    b_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # define the tensor-based indices
    pid = tl.program_id()
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    # load the input
    a_values = tl.load(a_ptr + offsets, mask=mask)
    # compute the cosine
    b_values = tl.cos(a_values)
    # store the output
    tl.store(b_ptr + offsets, b_values, mask=mask)

def cos(a, b):
    # define the grid size
    n_elements = a.num_elements
    BLOCK_SIZE = 2**(tl.ceil(tl.log2(n_elements))).to(tl.int32)
    grid = lambda meta: (meta.n_threads // BLOCK_SIZE, )
    # execute the kernel
    cos_func[grid](a.ptr, b.ptr, n_elements, BLOCK_SIZE)
