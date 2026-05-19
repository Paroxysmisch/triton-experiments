import torch
import triton
import triton.language as tl

@triton.jit
def floor_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    
    y = tl.math.floor(x)
    tl.store(y_ptr + offsets, y, mask=mask)


def floor(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    if x.is_floating_point():
        pass
    elif x.is_integer():
        return x.clone()
    else:
        raise RuntimeError("only floating point and integer types are supported")

    if out is None:
        out = torch.empty_like(x)
    else:
        assert x.size() == out.size() and x.stride() == out.stride()

    assert out is not x

    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    floor_kernel[grid](x, out, n_elements, BLOCK_SIZE=1024)
    return out
