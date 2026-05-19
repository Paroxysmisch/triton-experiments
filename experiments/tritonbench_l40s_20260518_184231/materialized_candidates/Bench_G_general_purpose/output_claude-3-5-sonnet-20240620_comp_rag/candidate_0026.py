import torch
import triton
import triton.language as tl
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice

# ... existing imports ...

@triton.autotune(
    configs=[
        triton.Config({"XBLOCK": 1, "RBLOCK": 1024}, num_stages=1, num_warps=8),
        triton.Config({"XBLOCK": 1, "RBLOCK": 2048}, num_stages=1, num_warps=8),
    ],
    key=["xnumel", "rnumel"],
)
@triton.jit
def triton_red_fused_native_layer_norm_no_welford(
    in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, out_ptr0,
    xnumel, rnumel, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr,
):
    # Grid and block setup
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex

    # First pass: compute mean
    _tmp3 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (rnumel * x0)), rmask).to(tl.float32)
        _tmp3 = _tmp3 + tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    
    # Calculate mean
    tmp3 = tl.sum(_tmp3, 1)[:, None]
    mean = tmp3 / rnumel
    tl.store(in_out_ptr0 + x0, mean, None)

    # Second pass: compute variance
    _tmp12 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp7 = tl.load(in_ptr0 + (r1 + (rnumel * x0)), rmask).to(tl.float32)
        tmp9 = tmp7 - mean
        _tmp12 = _tmp12 + tl.broadcast_to(tmp9 * tmp9, [XBLOCK, RBLOCK])

    # Calculate variance and inverse std
    var = tl.sum(_tmp12, 1)[:, None] / rnumel
    inv_std = libdevice.rsqrt(var + 1e-5)
    tl.store(in_out_ptr1 + x0, inv_std, None)

    # Final pass: normalize
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        
        # Load inputs
        x = tl.load(in_ptr0 + (r1 + (rnumel * x0)), rmask).to(tl.float32)
        gamma = tl.load(in_ptr1 + r1, rmask).to(tl.float32)
        beta = tl.load(in_ptr2 + r1, rmask).to(tl.float32)
        
        # Normalize
        x_norm = (x - mean) * inv_std
        out = x_norm * gamma + beta
        
        # Store result
        tl.store(out_ptr0 + (r1 + (rnumel * x0)), out, rmask)

def fused_native_layer_norm_no_welford(primals_1, primals_2, primals_3):
    # Get input dimensions
    S, D = primals_3.shape
    
    # Allocate output buffers
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        mean_buf = torch.empty((S, 1), dtype=torch.float32, device='cuda')
        inv_std_buf = torch.empty((S, 1), dtype=torch.float32, device='cuda')
        out_buf = torch.empty((S, D), dtype=torch.bfloat16, device='cuda')
        
        # Launch kernel
        grid = lambda META: (triton.cdiv(S, META["XBLOCK"]),)
        triton_red_fused_native_layer_norm_no_welford[grid](
            mean_buf, inv_std_buf, primals_3, primals_1, primals_2, 
            out_buf, S, D
        )
    
    return (out_buf, primals_3, mean_buf, inv_std_buf)
