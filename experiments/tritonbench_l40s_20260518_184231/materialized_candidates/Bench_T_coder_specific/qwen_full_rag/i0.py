import torch
import triton
import triton.language as tl


@triton.jit
def i0_kernel(x, out, N, BLOCK_SIZE: tl.constexpr):
    # Convert thread index to int
    pid = tl.program_id(0).to(tl.int64)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < N

    # Load input data
    x = tl.load(x + offset, mask=mask)

    # Initialize output
    out_val = tl.zeros_like(x)
    c = tl.zeros_like(x)
    c0 = tl.ones_like(x)
    k = tl.zeros_like(x, dtype=tl.int32)

    # Constants
    TWO = tl.full_like(k, 2)
    FOUR = tl.full_like(k, 4)

    # Iterative computation for the series expansion
    while True:
        c1 = c0 * ((k * k + 1) // (TWO * k + 1))
        c = tl.where(mask & (k == 0), c0, tl.where(mask & (k >= 2), c1, c))

        cur_term = tl.pow((x / 2), k) * c
        out_val += cur_term
        prev_val = tl.load(out + offset, mask=mask)
        out_val = tl.where(mask & (tl.abs(cur_term) > tl.abs(prev_val)), out_val, prev_val)

        k += 1
        cond = tl.any(mask & (tl.abs(cur_term) > tl.abs(prev_val)))
        if not cond:
            break

    # Store final results
    tl.store(out + offset, out_val, mask=mask)


def i0(inp, *, out=None):
    inp = inp.contiguous()
    if out is None:
        out = torch.empty_like(inp)
    else:
        out = out.contiguous()

    N = inp.numel()
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']), )
    i0_kernel[grid](inp, out, N, BLOCK_SIZE=1024)
    return out
