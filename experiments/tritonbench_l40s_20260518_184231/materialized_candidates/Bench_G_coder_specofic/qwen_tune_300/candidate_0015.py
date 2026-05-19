import torch
import triton
import triton.language as tl

def broadcastable(shape1, shape2):
    """check if two shapes are broadcastable"""
    return all(s1 == s2 or s1 == 1 or s2 == 1 for s1, s2 in zip(shape1, shape2))

@triton.jit
def cfggen():
    block_m = [1, 2, 4]
    block_n = [1024, 2048, 4096]
    warps = [4, 8, 16]
    configs = [
        triton.Config({"BLOCK_SIZE": m * n}, num_warps=w)
        for m in block_m
        for n in block_n
        for w in warps
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
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    inp = tl.load(inp_ptr + offset, mask=mask)
    select_mask = tl.load(select_mask_ptr + offset, mask=mask).to(tl.int1)
    out_offset = tl.load(prefix_sum_ptr + offset, mask=mask)

    tl.store(out_ptr + out_offset, inp, mask=(select_mask and mask))

def masked_select(inp, mask):
    inp_shape = tuple(inp.shape)
    mask_shape = tuple(mask.shape)

    assert broadcastable(
        inp_shape, mask_shape
    ), "The shapes of the `mask` and the `input` tensor must be broadcastable"
    assert len(
        mask_shape
    ) <= 3, "The `mask` tensor must have at most 3 dimensions, but got {}".format(
        len(mask_shape)
    )

    inp, mask = torch.broadcast_tensors(inp, mask)

    mask_flattened = mask.reshape(-1)
    inp_flattened = inp.reshape(-1)

    n_elements = inp.numel()
    prefix_sum = mask_flattened.cumsum(axis=0)
    output = torch.empty(prefix_sum[-1].item(), dtype=inp.dtype, device=inp.device)

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    with torch.cuda.device(inp.device):
        masked_select_kernel[grid](inp_flattened, mask_flattened, prefix_sum, output)
    return output
