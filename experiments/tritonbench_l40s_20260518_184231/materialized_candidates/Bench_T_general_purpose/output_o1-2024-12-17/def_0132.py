import triton
import triton.language as tl
import torch

@triton.jit
def _mul_sub_kernel(
    input_ptr, other_mul_ptr, other_sub_ptr, out_ptr,
    alpha, n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(other_mul_ptr + offsets, mask=mask, other=0.0)
    z = tl.load(other_sub_ptr + offsets, mask=mask, other=0.0)

    out_val = (x * y) - alpha * z
    tl.store(out_ptr + offsets, out_val, mask=mask)

def mul_sub(input, other_mul, other_sub, alpha=1, out=None):
    if not isinstance(input, torch.Tensor):
        input = torch.tensor(input, dtype=torch.float32)
    if not isinstance(other_mul, torch.Tensor):
        other_mul = torch.tensor(other_mul, dtype=torch.float32)
    if not isinstance(other_sub, torch.Tensor):
        other_sub = torch.tensor(other_sub, dtype=torch.float32)

    broadcast_shape = torch.broadcast_shapes(input.shape, other_mul.shape, other_sub.shape)
    input_b = input.expand(broadcast_shape)
    other_mul_b = other_mul.expand(broadcast_shape)
    other_sub_b = other_sub.expand(broadcast_shape)

    if out is None:
        out = torch.empty_like(input_b)

    input_b_flat = input_b.contiguous().view(-1)
    other_mul_b_flat = other_mul_b.contiguous().view(-1)
    other_sub_b_flat = other_sub_b.contiguous().view(-1)
    out_flat = out.contiguous().view(-1)
    
    n_elements = out_flat.numel()
    BLOCK_SIZE = 1024
    grid = lambda META: ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    _mul_sub_kernel[grid](
        input_b_flat, 
        other_mul_b_flat, 
        other_sub_b_flat,
        out_flat, 
        alpha,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return out
