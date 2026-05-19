import torch
import triton
import triton.language as tl
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice

empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch.ops.inductor._reinterpret_tensor

@triton.autotune(
    configs=[
        triton.Config(
            {
                "XBLOCK": 1,
                "RBLOCK": 1024,
            },
            num_stages=1,
            num_warps=8,
        ),
        triton.Config(
            {
                "XBLOCK": 1,
                "RBLOCK": 2048,
            },
            num_stages=1,
            num_warps=8,
        ),
    ],
    key=["xnumel", "rnumel"],
)
@triton.jit
def triton_red_fused_native_layer_norm_no_welford(
    in_out_ptr0,
    in_out_ptr1,
    in_ptr0,
    in_ptr1,
    in_ptr2,
    out_ptr0,
    xnumel,
    rnumel,
    XBLOCK: tl.constexpr,
    RBLOCK: tl.constexpr,
):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex

    # Initialize accumulators for mean and variance
    sum_val = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    sum_sq_val = tl.zeros([XBLOCK, RBLOCK], tl.float32)

    # Compute mean
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(
            in_ptr0 + (r1 + (rnumel * x0)), rmask, eviction_policy="evict_last"
        ).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
        sum_val += tmp2

    mean = tl.sum(sum_val, 1)[:, None] / rnumel
    tl.store(in_out_ptr0 + (x0), mean, None)

    # Compute variance
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(
            in_ptr0 + (r1 + (rnumel * x0)), rmask, eviction_policy="evict_last"
        ).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tmp1 - mean
        tmp3 = tmp2 * tmp2
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
        sum_sq_val += tmp4

    variance = tl.sum(sum_sq_val, 1)[:, None] / rnumel
    inv_std = libdevice.rsqrt(variance + 1e-05)
    tl.store(in_out_ptr1 + (x0), inv_std, None)

    # Normalize and apply scale and shift
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(
            in_ptr0 + (r1 + (rnumel * x0)), rmask, eviction_policy="evict_first"
        ).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tmp1 - mean
        tmp3 = tmp2 * inv_std
        tmp4 = tl.load(in_ptr1 + (r1), rmask, eviction_policy="evict_last").to(tl.float32)
        tmp5 = tmp3 * tmp4
        tmp6 = tl.load(in_ptr2 + (r1), rmask, eviction_policy="evict_last").to(tl.float32)
        tmp7 = tmp5 + tmp6
        tl.store(out_ptr0 + (r1 + (rnumel * x0)), tmp7, rmask)

def fused_native_layer_norm_no_welford(primals_1, primals_2, primals_3):
    S, D = primals_3.shape
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((S, 1), (1, S), torch.float32)
        buf1 = buf0
        del buf0  # reuse
        buf2 = empty_strided_cuda((S, 1), (1, S), torch.float32)
        buf3 = reinterpret_tensor(buf2, (S, 1), (1, 1), 0)
        del buf2  # reuse
        buf4 = empty_strided_cuda((S, D), (D, 1), torch.bfloat16)
        stream0 = get_raw_stream(0)
        grid = lambda META: (triton.cdiv(S, META["XBLOCK"]),)
        triton_red_fused_native_layer_norm_no_welford[grid](
            buf1, buf3, primals_3, primals_1, primals_2, buf4, S, D
        )
    return (
        buf4,
        primals_3,
        buf1,
        buf3,
    )
