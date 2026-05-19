import torch
import triton
import triton.language as tl
from .act_kernels import selu_forward
from .utils import calculate_settings_2d

class FusedInstanceNormSELUConv2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, num_features=None,
                eps=1e-5, momentum=0.1, affine=False, track_running_stats=False):
        ctx.save_for_backward(x)
        ctx.x_shape = x.shape
        x = x.contiguous()

        has_groups = groups != 1
        has_bias = bias is not None
        use_num_features = num_features is not None and num_features > 0

        if has_groups:
            assert x.shape[1] % groups == 0, "Number of channels must be divisible by number of groups"
        else:
            groups = x.shape[1]

        out = torch.empty_like(x)
        N, C, H, W = x.shape
        num_features = C if use_num_features else None

        if isinstance(stride, int):
            stride_h = stride_w = stride
        elif isinstance(stride, (list, tuple)):
            stride_h, stride_w = stride
        else:
            raise ValueError("Invalid stride")

        if isinstance(dilation, int):
            dilation_h = dilation_w = dilation
        elif isinstance(dilation, (list, tuple)):
            dilation_h, dilation_w = dilation
        else:
            raise ValueError("Invalid dilation")

        kernel_size = weight.shape[-2:]
        padding_h, padding_w = calculate_settings_2d(
            dim_size=H,
            kernel_size=kernel_size[0],
            stride=stride_h,
            dilation=dilation_h,
            padding=padding,
            ceil_mode=True
        )[:2]

        meta = {
            'num_warps': 4,
            'num_stages': 2,
        }
        rms_norm_weight = weight.flatten(1).contiguous()
        rms_norm_eps = eps
        trmsnorm(
            x, rms_norm_weight, rms_norm_eps,
            M=N, N=C, K=H*W,
            out=out,
            _indirection=x.stride(0),
            _V=x.stride(1),
            _b=x.stride(2),
            _S=H*W,
            BLOCK_N=meta['num_warps'],
            BLOCK_K=1,
            **meta
        )

        out_grad_fn = getattr(out, "grad_fn", None)
        assert out_grad_fn is not None, "Out tensor must be a result of a graph computation"
        assert out_grad_fn.name == "RMSNormBackward", "Top most op should be rms norm"

        return out

def fused_instance_norm_selu_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, num_features=None, eps=1e-5, momentum=0.1, affine=False, track_running_stats=False) -> torch.Tensor:
    """
    Applies a 2D convolution, SELU activation, and Instance Normalization on the input tensor.

    Args:
        input (Tensor): Input tensor of shape (minibatch, in_channels, iH, iW).
        weight (Tensor): Weights for the convolution, shape (out_channels, in_channels / groups, kH, kW).
        bias (Tensor, optional): Bias for the convolution layer, shape (out_channels).
        stride (int or tuple, optional): Stride of the convolution. Default is 1.
        padding (int or tuple, optional): Padding for the convolution. Default is 0.
        dilation (int or tuple, optional): Spacing between kernel elements. Default is 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Default is 1.
        num_features (int, optional): Number of features or channels in the input for instance normalization.
        eps (float, optional): A value added to the denominator for numerical stability in instance normalization. Default is 1e-5.
        momentum (float, optional): Momentum for updating running statistics in instance normalization. Default is 0.1.
        affine (bool, optional): If True, instance normalization has learnable affine parameters. Default is False.
        track_running_stats (bool, optional): If True, tracks running mean and variance for instance normalization. Default is False.

    Returns:
        Tensor: Output tensor after performing the fused operations.
    """
    return FusedInstanceNormSELUConv2d.apply(input, weight, bias, stride, padding, dilation, groups, num_features, eps, momentum, affine, track_running_stats)
