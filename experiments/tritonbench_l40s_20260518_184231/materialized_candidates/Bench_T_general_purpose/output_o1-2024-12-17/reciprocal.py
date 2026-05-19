import triton
import triton.language as tl

@triton.jit
def _reciprocal_kernel(in_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr + offsets, mask=mask)
    x = x.to(tl.float32)
    reciprocal_val = 1.0 / x
    tl.store(out_ptr + offsets, reciprocal_val, mask=mask)

def reciprocal(input, *, out=None):
    if not input.is_floating_point():
        input = input.to_default_float()  # Promote integral types to default float
    if out is None:
        out = input.empty_like()

    n_elements = input.numel()
    # Grid settings
    BLOCK_SIZE = 1024
    grid = lambda meta: ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    _reciprocal_kernel[grid](
        input,
        out,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return out
