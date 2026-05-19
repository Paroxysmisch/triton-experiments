import triton
import triton.language as tl

@triton.jit
def triton_red_fused_native_layer_norm_no_welford(
    in_ptr0, in_ptr1, in_ptr2, out_ptr0, in_out_ptr0, in_out_ptr1,
    xnumel, rnumel, eps: tl.float32, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * XBLOCK
    offsets = block_start + tl.arange(0, XBLOCK)
    mask = offsets < xnumel
    x = tl.load(in_ptr0 + offsets, mask=mask, other=0.0).to(tl.float32)

    # Compute mean
    mean = tl.sum(x, axis=0) / rnumel

    # Compute variance
    x_minus_mean = x - mean
    var = tl.sum(x_minus_mean * x_minus_mean, axis=0) / rnumel

    # Compute inverse standard deviation
    inv_std = tl.libdevice.rsqrt(var + eps)

    # Store mean and inv_std
    tl.store(in_out_ptr0 + pid, mean)
    tl.store(in_out_ptr1 + pid, inv_std)

    # Normalize and apply scale and shift
    normalized = (x - mean) * inv_std
    scale = tl.load(in_ptr1 + offsets, mask=mask, other=1.0).to(tl.float32)
    shift = tl.load(in_ptr2 + offsets, mask=mask, other=0.0).to(tl.float32)
    out = normalized * scale + shift

    # Store the result
    tl.store(out_ptr0 + offsets, out, mask=mask)

import torch
import triton
import triton.language as tl

def fused_native_layer_norm_no_welford(x, scale, shift, eps=1e-5):
    # Ensure inputs are on the same device
    device = x.device
    assert x.device == scale.device == shift.device, "All inputs must be on the same device"

    # Get the number of elements
    xnumel = x.numel()
    rnumel = x.shape[-1]

    # Allocate output buffers
    out = torch.empty_like(x)
    mean = torch.empty((xnumel // rnumel,), dtype=torch.float32, device=device)
    inv_std = torch.empty((xnumel // rnumel,), dtype=torch.float32, device=device)

    # Define the grid and block sizes
    grid = (xnumel // rnumel,)

    # Define the autotuning configurations
    triton_config = triton.Config({'XBLOCK': 128, 'RBLOCK': 32})
    triton_config_2 = triton.Config({'XBLOCK': 256, 'RBLOCK': 64})

    # Launch the kernel
    triton_red_fused_native_layer_norm_no_welford[grid](
        x, scale, shift, out, mean, inv_std,
        xnumel, rnumel, eps,
        XBLOCK=triton_config.XBLOCK, RBLOCK=triton_config.RBLOCK
    )

    return out, mean, inv_std
