import torch
import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

# Triton kernel for masked select operation
@triton.jit
def masked_select_kernel(
    inp_ptr, select_mask_ptr, prefix_sum_ptr, out_ptr, n_elements, **meta
):
    TMASK = meta["tl_types"]["mask"]
    tid = tl.program_id(0)
    tl_offsets = tl.arange(0, 1)
    if tl_offsets + tid * BLOCK_SIZE < n_elements:
        r = tl.load(
            inp_ptr + tl_offsets + tid * BLOCK_SIZE,
            mask=(tl_offsets + tid * BLOCK_SIZE) < n_elements,
            other=0.0,
        )
        m = tl.load(
            select_mask_ptr + tl_offsets + tid * BLOCK_SIZE,
            mask=(tl_offsets + tid * BLOCK_SIZE) < n_elements,
            other=0.0,
        )
        out_position = tl.load(
            prefix_sum_ptr + tl_offsets + tid * BLOCK_SIZE,
            mask=(tl_offsets + tid * BLOCK_SIZE) < n_elements,
            other=0.0,
        )
        # NOTE: we can't use `tl.where` here because that wouldn't short-circuit
        # and we'd write to memory locations even if the condition is false.
        # Use a local variable to work around this.
        should_store = (m != False)  # noqa: E712, E711
        if should_store != 0:
            tl.store(
                out_ptr + out_position - 1,
                r,
                mask=(out_position - 1) < n_elements,
            )

# Wrapper function for masked select operation
def masked_select(inp, mask, *, from_tensor=None):
    broadcastable(inp.shape, mask.shape)
    mask = mask.to(torch.bool)
    flattened_mask = mask.ravel()
    prefix_sum = flattened_mask.cumsum(axis=0)
    out = torch.empty(
        (triton_helpers.next_power_of_2(prefix_sum[-1].item()),),
        dtype=inp.dtype,
        device=inp.device,
    )
    n_elements = out.size(0)
    meta = {"num_warps": 1, "num_stages": 2}
    grid_fn = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    with torch.cuda.device(inp.device):
        masked_select_kernel[grid_fn](
            inp, mask, prefix_sum, out, n_elements, **meta
        )
    return out[: prefix_sum[-1]]

# Autotuning function for masked select operation
@config_of_constants(
    {
        "BLOCK_SIZE": [256, 512, 1024, 2048, 4096],
        "num_warps": [2, 4, 8],
    }
)
def cfggen():
    def inner(
        n_elements,
    ):
        return {"num_stages": 1 + int(triton.cdiv(n_elements, BLOCK_SIZE))}

    return inner

# Utility function for broadcasting tensor shapes
def broadcastable(shape1, shape2):
    shape1 = list(shape1)
    shape2 = list(shape2)

    if len(shape1) < len(shape2):
        shape1, shape2 = shape2, shape1

    for i in range(len(shape1) - len(shape2)):
        shape2.insert(0, 1)

    for i in range(len(shape1)):
        if shape1[i] != shape2[i] and min(shape1[i], shape2[i]) != 1:
            raise RuntimeError("shapes don't match: {} (and {})".format(shape1, shape2))

    return (shape1, shape2)


class MaskedSelect(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, mask):
        out = masked_select(inp, mask)
        ctx.save_for_backward(inp, mask)
        return out

    @staticmethod
    def backward(ctx, out_grad):
        (inp, mask) = ctx.saved_tensors
        inp_grad = torch.zeros_like(inp)
        masked_select_inp_grad = MaskedSelect.apply(inp, mask)
        inp_grad = inp_grad + masked_select_inp_grad
        return inp_grad, None


class MaskedSelectPlugin(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, mask):
        out = masked_select(inp, mask)
        ctx.save_for_backward(inp, mask)
        return out

    @staticmethod
    def backward(ctx, out_grad):
        (inp, mask) = ctx.saved_tensors
        inp_grad = torch.zeros_like(inp)
        masked_select_inp_grad = MaskedSelectPlugin.apply(inp, mask)
        inp_grad = inp_grad + masked_select_inp_grad
        return inp_grad, None
