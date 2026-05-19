import torch
import triton
import triton.language as tl
from torch import Tensor
from itertools import product

def cfggen():
    block_args = [16, 32, 64, 128, 256, 512, 1024]
    warps_args = [2, 4, 8, 16, 32]
    configs = [
        triton.Config({"XBLOCK": block}, num_warps=warp)
        for block, warp in product(block_args, warps_args)
    ]
    return configs

@triton.jit
def masked_select_kernel(
    inp_ptr,
    select_mask_ptr,
    prefix_sum_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    inp = tl.load(inp_ptr + offsets, mask=mask, other=0.0)
    select_mask = tl.load(select_mask_ptr + offsets, mask=mask, other=0.0).to(tl.int1)
    out_offset = tl.load(prefix_sum_ptr + offsets, mask=mask, other=0.0)

    tl.store(out_ptr + out_offset, inp, mask=(select_mask and mask))

def masked_select(inp: Tensor, mask: Tensor) -> Tensor:
    inp_shape = tuple(inp.shape)
    mask_shape = tuple(mask.shape)

    if broadcastable(inp_shape, mask_shape):
        inp, mask = torch.broadcast_tensors(inp, mask)
    else:
        raise RuntimeError("The shapes of the `mask` and the `input` tensor must be broadcastable")

    mask_flattened = mask.contiguous().view(-1)
    inp_flattened = inp.contiguous().view(-1)

    n_elements = inp.numel()
    prefix_sum = mask_flattened.cumsum(axis=0)
    out = torch.empty(prefix_sum[-1].item(), dtype=inp.dtype, device=inp.device)

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    masked_select_kernel[grid](inp_flattened, mask_flattened, prefix_sum, out, n_elements)
    return out

def broadcastable(shape1, shape2):
    return all(s1 == s2 or s1 == 1 or s2 == 1 for s1, s2 in zip(shape1, shape2))
