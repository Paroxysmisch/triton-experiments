import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.utils import maybe_profile
from torch._inductor.codecache import AsyncCompile
from torch._inductor import triton_helpers
from torch import empty_strided

@triton.autotune(
    configs=[
        triton.Config({"XBLOCK": 128, "RBLOCK": 64}, num_warps=4),
        triton.Config({"XBLOCK": 256, "RBLOCK": 128}, num_warps=8),
    ],
    key=["xnumel", "rnumel"],
)
@triton.jit
def triton_red_fused_native_layer_norm_no_welford(
    in_ptr0,
    in_ptr1,
    in_ptr2,
    in_out_ptr0,
    in_out_ptr1,
    out_ptr0,
    xnumel,
    rnumel,
    stride_in_out_0,
    stride_in_out_1,
    stride_out_0,
    XBLOCK: tl.constexpr,
    RBLOCK: tl.constexpr,
):
    """
    Triton kernel for layer normalization.
    """
    xnumel = xnumel
    rnumel = rnumel
    xoffset = tl.program_id(0) * XBLOCK
    roffset = tl.program_id(1) * RBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    rindex = roffset + tl.arange(0, RBLOCK)[None, :]

    xmask = xindex < xnumel
    rmask = rindex < rnumel

    x_before_r = XBLOCK // RBLOCK

    r_before_x = RBLOCK // XBLOCK

    x_after_r = x_before_r - 1
    r_after_x = r_before_x - 1

    x_after_r_mask = xindex >= XBLOCK * (xnumel // XBLOCK)
    r_after_x_mask = rindex >= RBLOCK * (rnumel // RBLOCK)

    x_loop_count = (xnumel + XBLOCK - 1) // XBLOCK
    r_loop_count = (rnumel + RBLOCK - 1) // RBLOCK

    mean0 = tl.zeros([XBLOCK], dtype=tl.float32)
    mean1 = tl.zeros([XBLOCK], dtype=tl.float32)

    for _ in range(r_loop_count):
        for _ in range(x_loop_count):
            x0 = tl.load(
                in_out_ptr0 + xindex * stride_in_out_0,
                mask=xmask & ~r_after_x_mask,
                other=0.0,
            )
            x1 = tl.load(
                in_out_ptr0 + xindex * stride_in_out_0,
                mask=xmask & ~r_after_x_mask,
                other=0.0,
            )
            mean0 += x0
            mean1 += x1

        mean0 = tl.sum(mean0, axis=0) / xnumel
        mean1 = tl.sum(mean1, axis=0) / xnumel

        for _ in range(r_loop_count):
            r0 = tl.load(
                in_out_ptr1 + rindex * stride_in_out_1,
                mask=rmask & ~x_after_r_mask,
                other=0.0,
            )
            r1 = tl.load(
                in_out_ptr1 + rindex * stride_in_out_1,
                mask=rmask & ~x_after_r_mask,
                other=0.0,
            )
            mean0 += r0
            mean1 += r1

        rmean0 = tl.sum(mean0, axis=0) / rnumel
        rmean1 = tl.sum(mean1, axis=0) / rnumel

        mean0 /= rnumel
        mean1 /= rnumel

        inv_std0 = tl.math.rsqrt(rmean0)
        inv_std1 = tl.math.rsqrt(rmean1)

        for _ in range(r_loop_count):
            r0 = tl.load(
                in_out_ptr1 + rindex * stride_in_out_1,
                mask=rmask & ~x_after_r_mask,
                other=0.0,
            )
            r1 = tl.load(
                in_out_ptr1 + rindex * stride_in_out_1,
                mask=rmask & ~x_after_r_mask,
                other=0.0,
            )
            x0 = tl.load(
                in_out_ptr0 + xindex * stride_in_out_0,
                mask=xmask & ~r_after_x_mask,
                other=0.0,
            )
            x1 = tl.load(
                in_out_ptr0 + xindex * stride_in_out_0,
                mask=xmask & ~r_after_x_mask,
                other=0.0,
            )
            r0 = (r0 - rmean0) * inv_std0
            r1 = (r1 - rmean1) * inv_std1
            x0 = (x0 - mean0) * inv_std0
            x1 = (x1 - mean1) * inv_std1
            tl.store(
                in_out_ptr0 + xindex * stride_in_out_0,
                x0,
                mask=xmask & ~r_after_x_mask,
            )
            tl.store(
                in_out_ptr0 + xindex * stride_in_out_0,
                x1,
                mask=xmask & ~r_after_x_mask,
            )
            tl.store(
                in_out_ptr1 + rindex * stride_in_out_1,
                r0,
                mask=rmask & ~x_after_r_mask,
            )
            tl.store(
                in_out_ptr1 + rindex * stride_in_out_1,
                r1,
                mask=rmask & ~x_after_r_mask,
            )

        mean0 = tl.zeros([XBLOCK], dtype=tl.float32)
        mean1 = tl.zeros([XBLOCK], dtype=tl.float32)

    for _ in range(r_loop_count):
        for _ in range(x_loop_count):
            x0 = tl.load(
                in_out_ptr0 + xindex * stride_in_out_0,
                mask=xmask & ~r_after_x_mask,
                other=0.0,
            )
            x1 = tl.load(
                in_out_ptr0 + xindex * stride_in_out_0,
                mask=xmask & ~r_after_x_mask,
                other=0.0,
            )
            mean0 += x0
            mean1 += x1

        mean0 = tl.sum(mean0, axis=0) / xnumel
        mean1 = tl.sum(mean1, axis=0) / xnumel

        for _ in range(r_loop_count):
            r0 = tl.load(
                in_out_ptr1 + rindex * stride_in_out_1,
                mask=rmask & ~x_after_r_mask,
                other=0.0,
            )
            r1 = tl.load(
                in_out_ptr1 + rindex * stride_in_out_1,
                mask=rmask & ~
