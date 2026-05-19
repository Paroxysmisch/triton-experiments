import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"num_warps": 1}),
        triton.Config({"num_warps": 2}),
        triton.Config({"num_warps": 4}),
        triton.Config({"num_warps": 8}),
        triton.Config({"num_warps": 16}),
        triton.Config({"num_warps": 32}),
    ],
    key=["group_size"],
)
@triton.jit
def _conv2d_st_fwd_kernel(
    input, weight, bias, output, stride, padding, dilation, group_size, N, C, H, W, kH, kW, _stride_n, _stride_c, _stride_h, _stride_w, _r_stride_n, _r_stride_c, _r_stride_h, _r_stride_w, NO_GROUPS: tl.constexpr, GROUP_SIZE: tl.constexpr, TILES_PER_GROUP: tl.constexpr, EVEN_K: tl.constexpr, BIAS: tl.constexpr, COMPLEX: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_HW: tl.constexpr
):
    # Kernel logic here...

def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    if groups != 1:
        raise NotImplementedError("Only groups == 1 is supported")
    if padding != 0:
        raise NotImplementedError("Only padding == 0 is supported")
    if dilation != 1:
        raise NotImplementedError("Only dilation == 1 is supported")

    N, C, H, W = input.shape
    out_channels, _, kH, kW = weight.shape
    stride_n, stride_c, stride_h, stride_w = input.stride()
    r_stride_n, r_stride_c, r_stride_h, r_stride_w = (
        weight.stride(0),
        weight.stride(1),
        weight.stride(2),
        weight.stride(3),
    )

    if bias is not None:
        bias = bias
    else:
        bias = torch.zeros((out_channels), device=input.device, dtype=input.dtype)

    if input.dtype == torch.complex64:
        input = input.float()
    if weight.dtype == torch.complex64:
        weight = weight.float()

    output = torch.empty((N, out_channels, H, W), device=input.device, dtype=input.dtype)
    grid = lambda META: (
        triton.cdiv(N, META["BLOCK_N"]),
        1,
        triton.cdiv(out_channels, META["GROUP_SIZE"]),
    )
    _conv2d_st_fwd_kernel[grid](
        input,
        weight,
        bias,
        output,
        stride,
        padding,
        dilation,
        groups,
        N,
        C,
        H,
        W,
        kH,
        kW,
        stride_n,
        stride_c,
        stride_h,
        stride_w,
        r_stride_n,
        r_stride_c,
        r_stride_h,
        r_stride_w,
        groups == 1,
        GROUP_SIZE=groups,
        TILES_PER_GROUP=4,
        BIAS=bias is not None,
        BLOCK_N=64,
        BLOCK_HW=16,
        NUM_SM=128,
        COMPLEX=input.dtype == torch.complex64 or weight.dtype == torch.complex64,
    )
    return output
