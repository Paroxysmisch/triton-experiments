import torch
import triton
import triton.language as tl


@triton.jit
def cos_kernel(x, y, n_elements, X_BLOCK_SIZE: tl.constexpr):
    idx = tl.program_id(axis=0)
    for i in range(idx, n_elements, X_BLOCK_SIZE):
        x[i] = tl.cos(y[i])


def cos(input, *, out=None):
    if out is None:
        out = torch.empty_like(input)
    else:
        assert out.is_empty()

    assert input.is_contiguous()
    N_ELEMENTS = input.numel()
    grid = lambda meta: (triton.cdiv(N_ELEMENTS, meta['X_BLOCK_SIZE']), )
    cos_kernel[grid](input, out, N_ELEMENTS, X_BLOCK_SIZE=1024)
    return out
