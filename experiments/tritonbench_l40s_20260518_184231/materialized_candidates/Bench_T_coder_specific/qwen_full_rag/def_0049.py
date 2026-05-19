import torch
import triton
import triton.language as tl
from .act_kernels import get_act_func

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 256}, num_stages=1),
    ],
    key=["height", "width"],
)
@triton.jit
def conv2d_leakyrelu_kernel(
    input_pointer,
    weight_pointer,
    bias_pointer,
    output_pointer,
    height,
    width,
    channels,
    n_elements,
    stride_input_batch,
    stride_input_height,
    stride_input_width,
    stride_input_channel,
    stride_weight_out,
    stride_weight_group,
    stride_weight_in,
    stride_weight_out_ch,
    stride_bias,
    stride_output_batch,
    stride_output_height,
    stride_output_width,
    stride_output_channel,
    GROUPS,
    BLOCK_SIZE: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr = 3,
    NORM: tl.constexpr = False,
    negative_slope: tl.constexpr = 0.01,
    REVERSE: tl.constexpr = False,
    PADDING: tl.constexpr = 0,
    DILATION: tl.constexpr = 1,
    INPLACE: tl.constexpr = False,
):
    pid_batch = tl.program_id(axis=0)
    pid_channel_out = tl.program_id(axis=1)
    # locate block
    batch_offset = pid_batch * stride_input_batch
    out_ch_off = pid_channel_out * stride_output_channel
    channel_offset = (pid_channel_out * GROUPS) * stride_weight_group

    block_offset_height = tl.arange(0, BLOCK_SIZE)
    block_offset_width = tl.arange(0, BLOCK_SIZE)

    input_block_ptr = (
        input_pointer
        + batch_offset
        + (PADDING + block_offset_height[:, None]) * stride_input_height
        + (PADDING + block_offset_width[None, :]) * stride_input_width
    )

    weight_block_ptr = (
        weight_pointer
        + channel_offset
        + stride_weight_group
        * (tl.arange(0, SLIDING_WINDOW) * stride_weight_in + block_offset_width)[None, :]
        + (tl.arange(0, SLIDING_WINDOW) * stride_weight_in + block_offset_height)[:, None]
        * stride_weight_height
    )
    input_tile = tl.load(
        input_block_ptr,
        mask=(block_offset_height[:, None] < height - PADDING)
        & (block_offset_width[None, :] < width - PADDING),
        other=0.0,
    )
    weight_tile = tl.load(weight_block_ptr)

    if NORM:
        norm_ptr = weight_pointer + channels * stride_weight_group + pid_channel_out
        norm = tl.load(norm_ptr).to(tl.float32)

    bias_tile = tl.load(bias_pointer + pid_channel_out * stride_bias)

    if REVERSE:
        weight_tile = tl.flip(weight_tile, [0])

    acc_tile = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)

    if DILATION > 1:
        input_dilated = tl.stride(input_tile, (stride_input_height, stride_input_width))
        for i in range(DILATION):
            curr_mask = (block_offset_height[:, None] == i) & (
                block_offset_width[None, :] == i
            )
            curr_view = tl.view(input_dilated, (BLOCK_SIZE, BLOCK_SIZE))
            acc_tile += tl.where(curr_mask, curr_view * weight_tile, 0.0)
    else:
        acc_tile += tl.dot(input_tile, weight_tile)

    if NORM:
        acc_tile /= norm

    acc_tile = acc_tile + bias_tile

    if not INPLACE:
        output_block_ptr = (
            output_pointer
            + pid_batch * stride_output_batch
            + out_ch_off
            + (PADDING + block_offset_height[:, None]) * stride_output_height
            + (PADDING + block_offset_width[None, :]) * stride_output_width
            + pid_channel_out * stride_output_channel
        )

        tl.store(
            output_block_ptr,
            acc_tile,
            mask=(block_offset_height[:, None] < height - PADDING)
            & (block_offset_width[None, :] < width - PADDING),
        )
    else:
        tl.store(
            input_block_ptr,
            acc_tile,
            mask=(block_offset_height[:, None] < height - PADDING)
            & (block_offset_width[None, :] < width - PADDING),
        )


def leaky_relu_conv2d(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
    negative_slope: float = 0.01,
    inplace: bool = False,
) -> torch.Tensor:
    assert input.is_contiguous()
    assert weight.is_contiguous()

    in_shape = input.shape
    weight_shape = weight.shape

    in_feature = in_shape[-1]
    out_feature = weight_shape[0]
    height = weight_shape[2]
    width = weight_shape[3]

    assert (
        in_feature % groups == 0
    ), "in_channels must be divisible by groups for grouped convolution"
    assert (
        out_feature % groups == 0
    ), "out_channels must be divisible by groups for grouped convolution"

    mid_dtype = (
        input.dtype
        if input.dtype in [torch.float16, torch.bfloat16, torch.float32]
        else torch.float32
    )

    n_groups = groups
    if len(in_shape) == 3:
        B, C, N = in_shape
        input = input.reshape(B * n_groups, -1, N)
    elif len(in_shape) == 4:
        B, C, H, W = in_shape
        input = input.reshape(B * n_groups, -1, H, W)
    else:
        raise ValueError("only accept 3D or 4D input")

    if bias is not None:
        assert bias.is_contiguous()
        assert bias.shape == (weight.shape[0],)
    assert weight.shape[0] == weight.shape[1], "only accept square filter"
    assert weight.shape[2] == weight.shape[3], "only accept square filter"

    sliding_window = weight.shape[2]
    output = torch.empty(
        (B, out_feature, in_shape[2], in_shape[3]),
        device=input.device,
        dtype=input.dtype,
    )

    grid = lambda META: (
        B,
        out_feature,
        triton.cdiv(in_shape[2], META["BLOCK_SIZE"]),
        triton.cdiv(in_shape[3], META["BLOCK_SIZE"]),
    )

    with torch.cuda.device(input.device):
        conv2d_leakyrelu_kernel[grid](
            input,
            weight,
            bias,
            output,
            in_shape[2],
            in_shape[3],
            in_feature,
            in_shape[-1],
            input.stride(0),
            input.stride(1),
            input.stride(2) if len(in_shape) == 4 else 1,
            input.stride(1),
            weight.stride(0),
            weight.stride(1),
            weight.stride(2),
            weight.stride(3),
            bias.stride(0) if bias is not None else 0,
            output.stride(0),
            output.stride(1),
            output.stride(2),
            output.stride(3),
            groups,
            NORM=False,
            negative_slope=negative_slope,
            REVERSE=False,
            PADDING=padding,
            DILATION=dilation,
            INPLACE=inplace,
        )
    return output


if __name__ == "__main__":
    from xformers.components import Activation
    from xformers.components.ops.swiglu import swiglu

    @torch.compile(fullgraph=True)
    def test_function(
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        w: torch.Tensor,
        act: Activation,
    ):
        if act == Activation.GeLU:
            return geformer_gelu(x, y, z, w, approximate="tanh")
        elif act == Activation.Swish:
            return geformer_swish(x, y, z, w)
        elif act == Activation.SIGMOID:
            return geformer_sigmoid(x, y, z, w)
        elif act == Activation.LiSwish:
            return geformer_liswish(x, y, z, w)
        elif act == Activation.HardShrink:
            return hardshrink_forward(x, y, z, w)
        elif act == Activation.HardTanh:
            return hardtanh_forward(x, y, z, w)
        elif act == Activation.QUANT_TANH:
            return quantize_tanh_forward(x, y, z, w)
        elif act == Activation.GELU_approx:
            return gelu_approx(x, y, z, w)
        elif act == Activation.SEGLU:
            return seggelu(x, y, z, w)
        elif act == Activation.SMERLU:
            return smelu(x, y, z, w)
        elif act == Activation.LeakyReLU:
            return leaky_relu_fwd(x, y, z, w)
        elif act == Activation.NONE:
            return swiglu(x, y, z, w)
        else:
            raise Exception(f"Unknown activation {act}")

    @torch.compile(fullgraph=True)
    def test_function_backward
