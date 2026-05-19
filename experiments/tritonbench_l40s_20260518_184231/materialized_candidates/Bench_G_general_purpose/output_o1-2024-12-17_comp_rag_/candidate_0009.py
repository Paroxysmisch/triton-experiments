import triton
import triton.language as tl
import torch

def broadcastable(shape1, shape2):
    return all(
        (s1 == s2 or s1 == 1 or s2 == 1)
        for s1, s2 in zip(shape1[::-1], shape2[::-1])
    )

def cfggen():
    configs = []
    for num_warps in [1, 2, 4]:
        for block_size in [128, 256, 512]:
            configs.append(
                triton.Config(
                    {'BLOCK_SIZE': block_size},
                    num_warps=num_warps
                )
            )
    return configs

@triton.autotune(configs=cfggen(), key=['n_elements'])
@triton.jit
def masked_select_kernel(
    inp_ptr, select_mask_ptr, prefix_sum_ptr, out_ptr,
    n_elements, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    inp_vals = tl.load(inp_ptr + offsets, mask=mask, other=0)
    mask_vals = tl.load(select_mask_ptr + offsets, mask=mask, other=0).to(tl.int1)
    out_positions = tl.load(prefix_sum_ptr + offsets, mask=mask, other=0)
    tl
