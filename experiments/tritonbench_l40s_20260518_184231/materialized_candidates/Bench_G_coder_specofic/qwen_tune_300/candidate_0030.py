import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch
from torch._inductor.runtime.triton_heuristics import persistent_kernel
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.kernel import Kernel
from torch._inductor import triton_helpers
from torch import libdevice

@triton.jit
@persistent_kernel(
    configs=[
        triton.Config({"XBLOCK": 128}, num_warps=4),
        triton.Config({"XBLOCK": 256}, num_warps=8),
    ],
    key=["xnumel", "rnumel"],
)
async def triton_red_fused_native_layer_norm_no_welford(
    in_out_ptr0,
    in_out_ptr1,
    in_ptr0,
    in_ptr1,
    in_ptr2,
    xnumel,
    rnumel,
    XBLOCK: tl.constexpr,
    RBLOCK: tl.constexpr,
):
    xoffset = tl.program_id(0) * XBLOCK
    rxoffset = tl.program_id(1) * RBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    rindex = rxoffset + tl.arange(0, RBLOCK)[None, :]
    xmask = xindex < xnumel
    rmask = rindex < rnumel

    # Load data
    x = tl.load(in_out_ptr0 + (xindex), xmask, other=0.0)
    if rnumel % RBLOCK == 0:
        r = tl.load(in_ptr0 + (rindex // RBLOCK), rmask, other=0.0)
    else:
        r = tl.load(in_ptr0 + (rindex), rmask, other=0.0)
    r = r.to(tl.float32)
    if rnumel % RBLOCK == 0:
        r = r + tl.arange(0, RBLOCK)[None, :]
    else:
        r = r + (rindex % RBLOCK)[None, :]
    r = tl.broadcast_to(r, (RBLOCK, XBLOCK))
    x = x * r

    # Compute mean
    mean = tl.sum(x, axis=0) / (xnumel * rnumel)

    # Accumulate into output
    tl.store(in_out_ptr1 + (tl.arange(0, XBLOCK)[:, None]), mean, xmask)

    # Compute variance
    x = tl.load(in_out_ptr0 + (xindex), xmask, other=0.0)
    x = tl.where(xmask, x - mean, 0.0)
    x = tl.where(rmask, x, 0.0)
    x = tl.where(rmask, x * x, 0.0)
    var = tl.sum(x, axis=0) / (xnumel * rnumel)
    rstd = libdevice.rsqrt(var)

    # Write-back mean and rstd
    tl.store(in_out_ptr1 + (xindex), mean, xmask)
    tl.store(in_out_ptr1 + (xindex + xnumel), rstd, xmask)

    # Normalize and apply linear transformation
    x = tl.load(in_out_ptr0 + (xindex), xmask, other=0.0)
    if rnumel % RBLOCK == 0:
        r = tl.load(in_ptr1 + (rindex // RBLOCK), rmask, other=0.0)
        b = tl.load(in_ptr2 + (rindex // RBLOCK), rmask, other=0.0)
    else:
        r = tl.load(in_ptr1 + (rindex), rmask, other=0.0)
        b = tl.load(in_ptr2 + (rindex), rmask, other=0.0)
    r = r.to(tl.float32)
    b = b.to(tl.float32)
    if rnumel % RBLOCK == 0:
        r = r + tl.arange(0, RBLOCK)[None, :]
        b = b + tl.arange(0, RBLOCK)[None, :]
    else:
        r = r + (rindex % RBLOCK)[None, :]
        b = b + (rindex % RBLOCK)[None, :]
    r = tl.broadcast_to(r, (RBLOCK, XBLOCK))
    b = tl.broadcast_to(b, (RBLOCK, XBLOCK))
    x = (x - mean) * rstd
    x = x * r + b

    # Write-back output
    tl.store(in_out_ptr0 + (xindex), x, xmask)


async def fused_native_layer_norm_no_welford(
    x, weight, bias, eps, out=None, input_is_normalized=False
):
    if out is None:
        out = torch.empty_like(x)
    else:
        assert x.shape == out.shape
    if x.stride(0) > 1 and x.stride(1) > 1:
        assert x.shape[1] < 65536, "This layer norm doesn't support feature dim >= 64KB"
    assert (
        weight is not None and bias is not None
    ), "This layer norm always uses weight and bias"
    assert x.is_contiguous()
    assert weight.is_contiguous()
    assert bias.is_contiguous()
    if x.ndim == 3:
        is_1d = True
        x = x.view(-1, x.shape[-1])
        is_1d = False
    else:
        assert x.ndim == 2
    stream = get_raw_stream(x.device.index)
    xnumel, rnumel = x.shape
    triton_red_fused_native_layer_norm_no_welford[(xnumel, rnumel // 128)](
        out,
        torch.empty_like(out, dtype=torch.float32),
        x,
        weight,
        bias,
        xnumel,
        rnumel,
        grid=(xnumel, rnumel // 128, 1),
        stream=stream,
    )
    return out
