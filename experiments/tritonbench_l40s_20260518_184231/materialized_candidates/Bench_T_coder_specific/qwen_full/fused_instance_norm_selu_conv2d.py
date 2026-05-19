import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.nn import functional as F

@triton.jit
def fused_instance_norm_selu_conv2d_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    z_ptr,
    mean_ptr,
    rstd_ptr,
    stride_x_n,
    stride_x_c,
    stride_x_h,
    stride_x_w,
    stride_z_n,
    stride_z_c,
    stride_z_h,
    stride_z_w,
    stride_w_g,
    stride_w_out,
    stride_w_in,
    stride_w_kh,
    stride_w_kw,
    stride_mean_g,
    stride_mean_out,
    stride_rstd_g,
    stride_rstd_out,
    N: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    g: tl.constexpr,
    SPATIAL_BLOCK_SIZE: tl.constexpr,
    GROUP_BLOCK_SIZE: tl.constexpr,
    SPLIT: tl.constexpr,
    BIAS: tl.constexpr,
    AFFINE: tl.constexpr,
    TRACK_RUNNING_STATS: tl.constexpr,
    NORM_BEFORE_ACTIVATION: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    # Kernel logic here...

def fused_instance_norm_selu_conv2d(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride: Union[int, Tuple[int, int]] = 1,
    padding: Union[int, Tuple[int, int]] = 0,
    dilation: Union[int, Tuple[int, int]] = 1,
    groups: int = 1,
    num_features: Optional[int] = None,
    eps: float = 1e-5,
    momentum: float = 0.1,
    affine: bool = False,
    track_running_stats: bool = False,
    nrom_before_activation: bool = False,
    activation: str = "selu",
) -> Tensor:
    if input.dtype == torch.float16:
        return F.instance_norm(
            input,
            weight,
            bias,
            running_mean=None,
            running_var=None,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )
    assert input.dtype == weight.dtype
    assert input.dim() == 4
    assert weight.dtype in [torch.float16, torch.float32]
    assert weight.dim() == 4
    assert weight.size(1) == input.size(1)
    assert input.size(0) == 1  # Minibatch must be 1
    assert weight.size(0) % groups == 0
    assert weight.size(1) % groups == 0
    assert input.size(1) == groups
    assert (bias is None) or (bias.size(0) == weight.size(0))
    assert 0 in [KH % 2 for KH in weight.shape[2:]]
    assert 0 in [KW % 2 for KW in weight.shape[2:]]
    if bias is not None:
        bias = bias.unsqueeze(0)
    if affine:
        affine = True
    if not track_running_stats:
        track_running_stats = False
    if num_features is None:
        num_features = input.size(1)
    else:
        assert num_features == input.size(1)
    N, C, H, W = input.shape
    g = groups
    out = torch.empty_like(input)
    mean = torch.empty((1, g, 1, 1), dtype=torch.float32, device=input.device)
    rstd = torch.empty((1, g, 1, 1), dtype=torch.float32, device=input.device)
    # Launcher
    def grid(meta):
        return (
            1,
            triton.cdiv(C, meta["GROUP_BLOCK_SIZE"]),
            triton.cdiv(H, meta["SPATIAL_BLOCK_SIZE"]),
        )
    fused_instance_norm_selu_conv2d_kernel[grid](
        input,
        weight,
        bias,
        out,
        mean,
        rstd,
        input.stride(0),
        input.stride(1),
        input.stride(2),
        input.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        weight.stride(0),
        weight.stride(1),
        weight.stride(2),
        weight.stride(3),
        mean.stride(0),
        mean.stride(1),
        rstd.stride(0),
        rstd.stride(1),
        N,
        C,
        H,
        W,
        weight.shape[2],
        weight.shape[3],
        g,
        SPATIAL_BLOCK_SIZE=8,
        GROUP_BLOCK_SIZE=triton.next_power_of_2(g),
        SPLIT=1,
        BIAS=bias is not None,
        AFFINE=affine,
        TRACK_RUNNING_STATS=track_running_stats,
        NORM_BEFORE_ACTIVATION=nrom_before_activation,
        ACTIVATION=activation,
        eps=eps,
        momentum=momentum,
        num_stages=1,
        num_warps=4,
    )
    return out
