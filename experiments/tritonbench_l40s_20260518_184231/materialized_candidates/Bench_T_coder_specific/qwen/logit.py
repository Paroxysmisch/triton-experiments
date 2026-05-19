import triton
import triton.language as tl

@triton.jit
def logit_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    eps,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    coords = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid_mask = coords < n_elements

    x = tl.load(input_ptr + coords, mask=valid_mask)

    # Clamp input values if eps is not None
    if eps != 0:
        x = tl.where(x < eps, eps, x)
        x = tl.where(x > 1 - eps, 1 - eps, x)

    # Compute logit
    z = x / (1 - x)
    y = tl.log(z)

    # Store result
    tl.store(output_ptr + coords, y, mask=valid_mask)

def logit(input, eps=None, out=None):
    input_shape = input.shape
    dtype = input.dtype

    if out is None:
        out = tl.zeros_like(input)

    assert input_shape == out.shape, "Input and output shapes must match"
    assert dtype in [tl.float32, tl.float64], "Unsupported data type"

    grid = lambda meta: (triton.cdiv(input.size, meta['BLOCK_SIZE']),)
    logit_kernel[grid](input.data, out.data, input.size, eps if eps is not None else 0.0, BLOCK_SIZE=256)

    return out
