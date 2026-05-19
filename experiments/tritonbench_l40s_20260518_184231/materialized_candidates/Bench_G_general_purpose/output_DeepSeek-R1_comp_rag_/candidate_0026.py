import torch
import triton
import triton.language as tl
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice

empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch.ops.inductor._reinterpret_tensor

@triton.autotune(
    configs=[
        triton.Config({"XBLOCK": 1, "RBLOCK": 1024}, num_stages=1, num_warps=8),
        triton.Config({"XBLOCK": 1, "RBLOCK": 2048}, num_stages=1, num_warps=8),
    ],
    key=["xnumel", "rnumel"],
)
@triton.jit
def triton_red_fused_native_layer_norm_no_welford(
    in_out_ptr0,  # Mean buffer
    in_out_ptr1,  # Inverse std buffer
    in_ptr0,       # Input data
    in_ptr1,       # Scale
    in_ptr2,       # Shift
    out_ptr0,      # Output
    xnumel,        # Size of the first dimension (S)
    rnumel,        # Size of the second dimension (D)
    XBLOCK: tl.constexpr,
    RBLOCK: tl.constexpr,
):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    
    x0 = xindex
    # First pass: Compute mean
    sum_acc = tl.zeros((XBLOCK, RBLOCK), dtype=tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        input_val = tl.load(in_ptr0 + (r1 + rnumel * x0), rmask, eviction_policy="evict_last").to(tl.float32)
        sum_acc += tl.where(rmask, input_val, 0.0)
    
    mean = tl.sum(sum_acc, axis=1) / rnumel
    tl.store(in_out_ptr0 + x0, mean[:, None], mask=xmask)
    tl.debug_barrier()
    
    # Second pass: Compute variance
    var_acc = tl.zeros((XBLOCK, RBLOCK), dtype=tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        input_val = tl.load(in_ptr0 + (r1 + rnumel * x0), rmask, eviction_policy="evict_last").to(tl.float32)
        diff = input_val - mean[:, None]
        var_acc += tl.where(rmask, diff * diff, 0.0)
    
    variance = tl.sum(var_acc, axis=1) / rnumel
    inv_std = libdevice.rsqrt(variance + 1e-05)
    tl.store(in_out_ptr1 + x0, inv_std[:, None], mask=xmask)
    tl.debug_barrier()
    
    # Third pass: Normalize and store
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        input_val = tl.load(in_ptr0 + (r1 + rnumel * x0), rmask, eviction_policy="evict_first").to(tl.float32)
        scale = tl.load(in_ptr1 + r1, rmask, eviction_policy="evict_last").to(tl.float32)
        shift = tl.load(in_ptr2 + r1, rmask, eviction_policy="evict_last").to(tl.float32)
        
        normalized = (input_val - mean[:, None]) * inv_std[:, None]
        scaled = normalized * scale
        shifted = scaled + shift
        tl.store(out_ptr0 + (r1 + rnumel * x0), shifted.to(input_val.dtype), rmask)

def fused_native_layer_norm_no_welford(scale, shift, input_tensor):
    S, D = input_tensor.shape
    device = input_tensor.device
    
    with torch.cuda.device(device):
        # Allocate buffers for mean and inverse std
        mean_buffer = empty_strided_cuda((S, 1), (1, S), torch.float32)
        inv_std_buffer = empty_strided_cuda((S, 1), (1, S), torch.float32)
        output = empty_strided_cuda((S, D), (D, 1), input_tensor.dtype)
        
        # Configure grid and launch kernel
        grid = lambda meta: (triton.cdiv(S, meta["XBLOCK"]),)
        triton_red_fused_native_layer_norm_no_welford[grid](
            mean_buffer, inv_std_buffer,
            input_tensor, scale, shift, output,
            S, D
        )
        
    return output, input_tensor, mean_buffer, inv_std_buffer
