import triton
import triton.language as tl
import torch

@triton.jit
def _seeded_dropout(
    x_ptr, 
    output_ptr, 
    n_elements, 
    p, 
    seed, 
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    rand_vals = tl.rand(seed, offsets)
    mask = rand_vals > p
    x = tl.load(x_ptr + offsets, mask=offsets < n_elements, other=0.0)
    out = x * mask * (1.0 / (1.0 - p))
    tl.store(output_ptr + offsets, out, mask=offsets < n_elements)

def seeded_dropout(x, p, seed=0):
    x_c = x.contiguous()
    out = torch.empty_like(x_c)
    BLOCK_SIZE = 1024
    grid = lambda meta: ((x_c.numel() + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    _seeded_dropout[grid](
        x_c, 
        out, 
        x_c.numel(), 
        p, 
        seed, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out
