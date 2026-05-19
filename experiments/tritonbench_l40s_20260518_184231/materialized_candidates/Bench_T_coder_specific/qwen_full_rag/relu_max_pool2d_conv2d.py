import torch
import triton
import triton.language as tl
from .utils import next_power_of_two
from .max_pool2d import max_pool2d
from .relu import relu
from .conv2d import conv2d


@triton.autotune(
    configs=[
        triton.Config({"num_warps": 2}),
        triton.Config({"num_warps": 4}),
        triton.Config({"num_warps": 8}),
    ],
    key=["nCHW", "kC"],
)
@triton.jit
def _relu_max_pool2d_conv2d_nchw(
    input, weight, bias, pre_act, nCHW, kC, sC, stride_n, stride_h, stride_w, stride_kh, stride_kw, padding_h, padding_w, dilation_h, dilation_w, stride_pool, pool_ph, pool_pw, pool_dilate_h, pool_dilate_w, ceil_mode, _, add_bias
):
    # get channel idx
    kh = tl.program_id(axis=0)
    kw = tl.program_id(axis=1)
    c = tl.program_id(axis=2)
    n_hw = tl.program_id(axis=3)
    # compute offset
    offset_n = n_hw // nCHW
    offset_hw = n_hw % nCHW
    offset_h = offset_hw // kC
    offset_w = offset_hw % kC

    # load input patch and weights
    # [hw, out_c, in_c, kh, kw]
    inp_patch_ptr = input + offset_n * stride_n + (offset_h * stride_h + dilation_h * offset_h) * stride_w + (offset_w * stride_w + dilation_w * offset_w) + c * sC
    w_ptr = weight + c * stride_kh * stride_kw + kh * stride_kh + kw * stride_kw
    # [in_c, out_c, kh, kw]
    inp_tensor = tl.load(inp_patch_ptr, mask=(c < sC) & (offset_h * stride_h + dilation_h * offset_h < stride_h) & (offset_w * stride_w + dilation_w * offset_w < stride_w)).to(tl.float32)
    w = tl.load(w_ptr, mask=(c < sC) & (kh < stride_kh) & (kw < stride_kw)).to(tl.float32)

    # [in_c]
    b = tl.zeros([1], dtype=tl.float32)
    if add_bias:
        b_ptr = bias + c
        b = tl.load(b_ptr, mask=c < sC).to(tl.float32)

    # [hw, out_c]
    res = tl.dot(inp_tensor, w) + b
    res = res.to(pre_act.dtype.element_ty)

    # apply activation and store
    act_out_ptr = pre_act + offset_n * nCHW * kC + offset_hw * kC + c
    tl.store(act_out_ptr, res)

    # apply pool
    def apply_pool(res, offset_h, offset_w):
        pool_src = pre_act + offset_n * nCHW * kC + ((offset_h * pool_dilate_h + pool_dilate_h) * kC + (offset_w * pool_dilate_w + pool_dilate_w)) + c
        pool_dest = (
            pre_act
            + offset_n * nCHW * kC
            + ((offset_h * stride_pool + pool_dilate_h) * kC + (offset_w * stride_pool + pool_dilate_w))
            + c
        )

        inp = tl.load(pool_src, mask=((offset_h * pool_dilate_h + pool_dilate_h < stride_h) & (offset_w * pool_dilate_w + pool_dilate_w < stride_w)))
        ref = tl.load(pool_dest, mask=((offset_h * stride_pool + pool_dilate_h < stride_h) & (offset_w * stride_pool + pool_dilate_w < stride_w)))
        out = tl.where(ref > inp, ref, inp)
        tl.store(pool_dest, out, mask=((offset_h * stride_pool + pool_dilate_h < stride_h) & (offset_w * stride_pool + pool_dilate_w < stride_w)))

    if pool_dilate_h != 0 or pool_dilate_w != 0:
        for hinc in range(0, tl.cdiv(stride_h, pool_dilate_h)):
            for winc in range(0, tl.cdiv(stride_w, pool_dilate_w)):
                apply_pool(res, hinc, winc)
    else:
        block_striding = triton.blocking_2d(num_programs=nCHW, num_stages=1, block_size=1, y_block_size=max(1, stride_pool), x_block_size=max(1, stride_pool))[::-1]

        grid = lambda meta: [meta["y"], meta["x"]]
        apply_pool[grid](res, 0, 0)

    relu_out_ptr = pre_act + offset_n * nCHW * kC + offset_hw * kC + c
    relu_out = tl.maximum(0, tl.load(relu_out_ptr))

    relu_dest = (
        pre_act
        + offset_n * nCHW * kC
        + ((offset_h * stride_pool + pool_dilate_h) * kC + (offset_w * stride_pool + pool_dilate_w))
        + c
    )
    tl.store(relu_dest, relu_out, mask=((offset_h * stride_pool + pool_dilate_h < stride_h) & (offset_w * stride_pool + pool_dilate_w < stride_w)))


def relu_max_pool2d_conv2d(
    input, weight, bias=None, conv_stride=1, conv_padding=0, conv_dilation=1, conv_groups=1, pool_kernel_size=2, pool_stride=None, pool_padding=0, pool_dilation=1, pool_ceil_mode=False, inplace=False
):
    """
    Apply conv2d, ReLU, and max_pool2d in a single layer. It's equivalent to the following operations being performed one after another:

        >>> x = F.conv2d(x, weight, bias=bias, stride=conv_stride, padding=conv_padding, dilation=conv_dilation, groups=conv_groups)
        >>> x = F.relu(x)
        >>> x = F.max_pool2d(x, pool_kernel_size, stride=pool_stride, padding=pool_padding, dilation=pool_dilation, ceil_mode=pool_ceil_mode)

    Args:
        input (Tensor): Input tensor of shape (N, C, H, W)
        weight (Tensor): Filter of shape (OutChannels, InChannels/Groups, K_H, K_W)
        bias (Tensor, optional): Bias of shape (OutChannels). Default: None
        conv_stride (int, optional): Stride of the convolution. Default: 1
        conv_padding (int, str, optional): Zero-padding added to both sides of the input. Default: 0
        conv_dilation (int, optional): Spacing between kernel elements. Default: 1
        conv_groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        pool_kernel_size (int, tuple): The size of the pooling window. Default: 2
        pool_stride (int, tuple, optional): The stride of the pooling window. Default: None (equal to the size of the pooling window)
        pool_padding (int, tuple, optional): The amount of implicit zero-padding added on both sides of the input. Default: 0
        pool_dilation (int, tuple, optional): The spacing between the kernel points. Default: 1
        pool_ceil_mode (bool, optional): Use ceil instead of floor to compute output shape. Default: False
        inplace (bool, optional): Whether to use inplace operation. Default: False

    Returns:
        Tensor: Output tensor
    """
    N, C, H, W = input.shape
    OutC, _, kH, kW = weight.shape
    stride_n = C * H * W
    stride_h = W
    stride_w = 1
    nCHW = N * H * W
    sC = C // conv_groups

    if isinstance(conv_padding, str):
        padding_h = padding_w = {
            "valid": (0, 0),
            "same": ((H - 1) // 2, H // 2),
        }[conv_padding.lower()]
    elif isinstance(conv_padding, (tuple, list)):
        padding_h, padding_w = conv_padding
    else:
        padding_h = padding_w = conv_padding

    if isinstance(pool_stride, (list, tuple)):
        stride_pool_h, stride_pool_w = pool_stride
    else:
        stride_pool_h = stride_pool_w = pool_stride

    if pool_stride is None:
        stride_pool_h = stride_pool_w = pool_kernel_size

    if isinstance(pool_padding, (list, tuple)):
        pool_pad_h, pool_pad_w = pool_padding
    else:
        pool_pad_h = pool_pad_w = pool_padding

    pool_dilate_h = pool_dilate_w = pool_dilation

    if not inplace:
        output = torch.empty(N, OutC, (H + 2 * padding_h - pool_dilate_h * (pool_kernel_size - 1) - 1) // stride_pool_h + 1, (W + 2 * padding_w - pool_dilate_w * (pool_kernel_size - 1) - 1) // stride_pool_w + 1, dtype=input.dtype, device=input.device)
    else:
