import torch
import triton
import triton.language as tl
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime.triton_helpers import libdevice

@triton.autotune(
    configs=[
        triton.Config({"XBLOCK": 1, "RBLOCK": 1024}, num_stages=1, num_warps=8),
        triton.Config({"XBLOCK": 1, "RBLOCK": 2048}, num_stages=1, num_warps=8),
    ],
    key=["xnumel", "rnumel"],
)
@triton.jit
def triton_red_fused_native_layer_norm_no_welford(
    in_out_ptr0,  # mean output
    in_out_ptr1,  # inv_std output
    in_ptr0,      # input tensor
    in_ptr1,      # weight
    in_ptr2,      # bias
    out_ptr0,     # output tensor
    xnumel,       # batch size
    rnumel,       # feature dimension
    XBLOCK: tl.constexpr,
    RBLOCK: tl.constexpr,
):
    # Grid and block setup
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex

    # Step 1: Compute mean
    _tmp3 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (rnumel * x0)), rmask).to(tl.float32)
        _tmp3 += tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    
    mean = tl.sum(_tmp3, 1)[:, None] / rnumel
    tl.store(in_out_ptr0 + x0, mean, None)
    
    # Step 2: Compute variance
    _var = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        val = tl.load(in_ptr0 + (r1 + (rnumel * x0)), rmask).to(tl.float32)
        diff = val - mean
        _var += tl.broadcast_to(diff * diff, [XBLOCK, RBLOCK])
    
    var = tl.sum(_var, 1)[:, None] / rnumel
    inv_std = libdevice.rsqrt(var + 1e-5)
    tl.store(in_out_ptr1 + x0, inv_std, None)

    # Step 3: Normalize and apply affine transform
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        
        # Load inputs
        val = tl.load(in_ptr0 + (r1 + (rnumel * x0)), rmask).to(tl.float32)
        weight = tl.load(in_ptr1 + r1, rmask).to(tl.float32)
        bias = tl.load(in_ptr2 + r1, rmask).to(tl.float32)
        
        # Normalize and transform
        normalized = (val - mean) * inv_std
        result = normalized * weight + bias
        
        # Store output
        tl.store(out_ptr0 + (r1 + (rnumel * x0)), result, rmask)

def fused_native_layer_norm_no_welford(primals_1, primals_2, primals_3):
    """
    Wrapper function for layer normalization without Welford's algorithm
    Args:
        primals_1: weights
        primals_2: bias
        primals_3: input tensor
    Returns:
        Tuple of (normalized tensor, input tensor, mean, inv_std)
    """
    S, D = primals_3.shape
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        
        # Allocate buffers
        mean_buf = torch.empty((S, 1), dtype=torch.float32, device='cuda')
        inv_std_buf = torch.empty((S, 1), dtype=torch.float32, device='cuda')
        output_buf = torch.empty_like(primals_3)
        
        # Launch kernel
        grid = lambda META: (triton.cdiv(S, META["XBLOCK"]),)
        triton_red_fused_native_layer_norm_no_welford[grid](
            mean_buf, inv_std_buf, primals_3, 
            primals_1, primals_2, output_buf, 
            S, D
        )
        
    return (output_buf, primals_3, mean_buf, inv_std_buf)
