import torch
import triton
import triton.language as tl

@triton.jit(do_not_specialize=["select_mask"])
def masked_select_kernel(
    inp_ptr,
    select_mask_ptr,
    prefix_sum_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    select_mask: tl.constexpr,  # If the mask is known at compile-time.
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    inp = tl.load(inp_ptr + offsets, mask=mask, other=0.0)
    prefix_sum = tl.load(prefix_sum_ptr + offsets, mask=mask)

    # Convert select_mask to Boolean type tl.int1.
    # Assume that if select_mask is given it must be broadcasted
    # if select_mask is known at compile-time.
    if tl.constexpr(select_mask is not None):
        select_mask = broadcastable(select_mask, inp)
    else:
        select_mask = broadcast_to(select_mask_ptr + offsets, inp)

    if select_mask.dtype is not tl.int1:
        select_mask = select_mask.to(tl.int1)

    out_pos = prefix_sum * select_mask

    tl.store(out_ptr + out_pos, inp, mask=(1 & mask & select_mask))

def cfggen():
    """
    Generate configurations for autotuning.

    `M` is the larger of the two, and `N` is the smaller.

    Generate configurations for the following autotune settings:
    1. {1024, 2048, 4096}-tile_n
    8-num_warps
    """
    configs = [
        triton.Config(
            {
                "BLOCK_SIZE": v,
            },
            num_warps=8,
        )
        for v in [1024, 2048, 4096]
    ]
    return configs

class MaskedSelect(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, mask, configs=None):
        if configs is None:
            configs = cfggen()

        BroadcastShapes(inp.shape, mask.shape)

        flattened_mask = mask.ravel()

        # Ensure we have a CUDA device.
        assert (
            inp.is_cuda and flattened_mask.is_cuda and inp.is_contiguous()
        ), "Only contiguous CUDA tensors are supported at the moment"

        # Flatten input and mask to 1D.
        n_elements = inp.numel()

        # Calculate the prefix sum of the mask to get the output positions.
        prefix_sum = torch.cumsum(flattened_mask, dim=0)

        # The output size is the number of True values in the mask.
        out = torch.empty(
            prefix_sum[-1].item(),
            dtype=inp.dtype,
            device=inp.device,
        )

        def grid(meta):
            return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

        # Call the Triton kernel with autotuning.
        with torch.cuda.device(inp.device):
            auto_config = triton.autotune(configs, key=["BLOCK_SIZE"])
            masked_select_kernel[grid](inp, flattened_mask, prefix_sum, out, n_elements)

        ctx.mark_non_differentiable(out)

        return out

def broadcastable(shape1, shape2):
    broadcast_shape = torch.Size(
        [
            max(s1, s2, d1, d2)
            for s1, d1, s2, s3, d2 in itertools.zip4(
                shape1,
                len(shape1) * [1],
                shape2,
                len(shape2) * [1],
            )
        ]
    )

    def broadcastable_broadcast_to(x: torch.Tensor, shape: torch.Size):
        out_shape = broadcast_shape[: -len(shape)] + shape
        return torch.broadcast_to(x, out_shape)

    return broadcastable_broadcast_to

def broadcast_to(broadcast_to, inp):
    broadcast_to_fn = broadcastable(inp.shape, broadcast_to.shape)
    return broadcast_to_fn(broadcast_to, inp.shape)
