import triton
import triton.language as tl


@triton.jit
def _logit_kernel(
    IN_PTR, OUT_PTR,
    N_ELEMENTS,
    EPS,
    USE_EPS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    start = pid * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N_ELEMENTS

    x = tl.load(IN_PTR + offsets, mask=mask)
    if USE_EPS == 1:
        # clamp x to [EPS, 1 - EPS]
        x = tl.where(x < EPS, EPS, x)
        x = tl.where(x > (1 - EPS), 1 - EPS, x)
        y = tl.log(x / (1 - x))
    else:
        # yield NaN if x < 0 or x > 1
        nan_mask = (x < 0.0) | (x > 1.0)
        x = tl.where(nan_mask, 0.5, x)  # placeholder to avoid log(0) and division by 0
        y = tl.log(x / (1.0 - x))
        y = tl.where(nan_mask, tl.nan, y)

    tl.store(OUT_PTR + offsets, y, mask=mask)


def logit(input, eps=None, *, out=None):
    # Assume 'input' and 'out' are Triton-compatible tensors with the same device placement
    if out is None:
        out = input.clone()  # or create a new empty tensor with the same shape/device

    n_elements = input.numel()
    BLOCK_SIZE = 1024
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    use_eps = 1 if eps is not None else 0
    eps_val = eps if eps is not None else 0.0

    _logit_kernel[grid](
        input, out,
        n_elements,
        eps_val,
        use_eps,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return out
