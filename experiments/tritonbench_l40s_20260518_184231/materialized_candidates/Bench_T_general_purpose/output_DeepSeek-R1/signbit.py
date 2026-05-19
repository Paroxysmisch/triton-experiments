import torch
import triton
import triton.language as tl

@triton.jit
def signbit_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    ELEMENT_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    if ELEMENT_SIZE == 1:
        dtype = tl.int8
    elif ELEMENT_SIZE == 2:
        dtype = tl.int16
    elif ELEMENT_SIZE == 4:
        dtype = tl.int32
    elif ELEMENT_SIZE == 8:
        dtype = tl.int64
    else:
        # Unsupported element size
        tl.static_assert(False, "Unsupported element size")

    # Load as signed integer; for floating points, this will correctly capture the sign bit via bitcast
    input_val = tl.load(input_ptr + offsets, mask=mask, other=0)
    input_int = tl.bitcast(input_val, dtype)
    sign = input_int < 0
    tl.store(output_ptr + offsets, sign, mask=mask)

def signbit(input, *, out=None):
    # Check if input is unsigned
    if not (input.dtype.is_floating_point or input.dtype.is_signed):
        if out is None:
            out = torch.zeros_like(input, dtype=torch.bool)
        else:
            out.fill_(False)
        return out

    element_size = input.element_size()
    if element_size not in [1, 2, 4, 8]:
        raise NotImplementedError(f"Unsupported element size: {element_size}")

    if out is None:
        out = torch.empty_like(input, dtype=torch.bool)

    n_elements = input.numel()
    if n_elements == 0:
        return out

    block_size = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    signbit_kernel[grid](
        input, out, n_elements,
        BLOCK_SIZE=block_size,
        ELEMENT_SIZE=element_size
    )
    return out
