import torch
import triton
import triton.language as tl

@triton.jit
def tanh_kernel(X, Y, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < N
    x = tl.load(X + offset, mask=mask)
    y = tl.libdevice.tanh(x)
    tl.store(Y + offset, y, mask=mask)

def tanh(input, *, out=None):
    assert input.is_contiguous(), "Only support contiguous input"
    if out == None:
        out = torch.empty_like(input)
    else:
        assert out.is_contiguous()
        assert out.shape == input.shape
        assert out.dtype == input.dtype

    N = input.numel()
    BLOCK_SIZE = triton.next_power_of_2(math.ceil(N ** 0.5))
    grid = (triton.cdiv(N, BLOCK_SIZE), )
    tanh_kernel[grid](input, out, N, BLOCK_SIZE)
    return out
