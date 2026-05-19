import torch
import triton
import triton.language as tl


@triton.jit
def conv2d_forward(
    input_pointer,
    weight_pointer,
    bias_pointer,
    other_pointer,
    output_pointer,
    input_batch_stride,
    input_in_channels_stride,
    input_height_stride,
    input_width_stride,
    weight_out_channels_stride,
    weight_in_channels_stride,
    weight_height_stride,
    weight_width_stride,
    output_batch_stride,
    output_out_channels_stride,
    output_height_stride,
    output_width_stride,
    alpha,
    PADDED_HEIGHT: tl.constexpr,
    PADDED_WIDTH: tl.constexpr,
    KERNEL_HEIGHT: tl.constexpr,
    KERNEL_WIDTH: tl.constexpr,
    GROUPS: tl.constexpr,
    INPUT_HEIGHT: tl.constexpr,
    INPUT_WIDTH: tl.constexpr,
    MBS: tl.constexpr,
    OCS: tl.constexpr,
    WCS: tl.constexpr,
    NUM_WEIGHT_ELEMENTS: tl.constexpr,
    TILE_HEIGHT: tl.constexpr,
    TILE_WIDTH: tl.constexpr,
):
    program_id = tl.program_id(axis=0)
    num_programs = tl.num_programs(axis=0)
    # compute batch and out_channel index
    batch_index = program_id // OCS
    remainder = program_id % OCS
    out_channel_index = remainder // MBS
    group_index = out_channel_index // GROUPS
    # set pointers to first element of each relevant batch, out_channel, group
    batch_offset = batch_index * input_batch_stride
    out_channel_offset = out_channel_index * weight_out_channels_stride
    group_offset = group_index * MBS

    input_pointer += batch_offset + out_channel_offset + group_offset
    weight_pointer += out_channel_offset + group_offset

    # set pointer to correct position in input and weight
    input_pointer += out_channel_index * input_in_channels_stride
    weight_pointer += out_channel_index * weight_in_channels_stride

    # all programs handle the same number of height and width tiles
    num_tiles_h = PADDED_HEIGHT // TILE_HEIGHT
    num_tiles_w = PADDED_WIDTH // TILE_WIDTH
    tile_index = tl.program_id(axis=1) + num_programs * tl.program_id(axis=2)
    tile_row_idx = tile_index // num_tiles_w
    tile_col_idx = tile_index % num_tiles_w

    input_tile_offset_y = tile_row_idx * TILE_HEIGHT * input_height_stride
    input_tile_offset_x = tile_col_idx * TILE_WIDTH * input_width_stride

    input_pointer += input_tile_offset_y + input_tile_offset_x

    weight_tile_offset_h = 0 * weight_height_stride
    weight_tile_offset_w = 0 * weight_width_stride

    weight_pointer += weight_tile_offset_h + weight_tile_offset_w

    acc_tile = tl.zeros((MBS, WCS), dtype=tl.int32)

    for h in range(KERNEL_HEIGHT):
        input_weight_tile_offset_y = (
            (TILE_HEIGHT - h - 1) * input_height_stride - h * weight_height_stride
        )

        for w in range(KERNEL_WIDTH):
            input_weight_tile_offset_x = (
                w * input_width_stride - w * weight_width_stride
            )

            current_input_pointer = (
                input_pointer
                + input_weight_tile_offset_y
                + input_weight_tile_offset_x
            )
            current_weight_pointer = (
                weight_pointer
                + input_weight_tile_offset_y
                + input_weight_tile_offset_x
            )

            input_chunk = tl.load(current_input_pointer, mask=True, other=0)
            weight_chunk = tl.load(current_weight_pointer)

            acc_tile += input_chunk * weight_chunk

    padded_tile_height = (tile_row_idx + 1) * TILE_HEIGHT
    padded_tile_width = (tile_col_idx + 1) * TILE_WIDTH

    acc_tile = tl.where(padded_tile_height <= INPUT_HEIGHT, acc_tile, 0)
    acc_tile = tl.where(padded_tile_width <= INPUT_WIDTH, acc_tile, 0)

    bias_pointer += out_channel_index
    bias_chunk = tl.load(bias_pointer)

    acc_tile += bias_chunk

    other_pointer += batch_offset + out_channel_offset + group_offset
    other_pointer += out_channel_index * other_pointer.stride(1)

    other_chunk = tl.load(other_pointer)
    acc_tile = acc_tile + other_chunk * alpha

    acc_tile = acc_tile.to(tl.float16)

    output_tile_offset_y = tile_row_idx * TILE_HEIGHT * output_height_stride
    output_tile_offset_x = tile_col_idx * TILE_WIDTH * output_width_stride

    output_pointer += (
        batch_offset
        + out_channel_offset
        + group_offset
        + output_tile_offset_y
        + output_tile_offset_x
    )

    tl.store(output_pointer, acc_tile, mask=False)


class Conv2dAdd(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input,
        weight,
        bias=None,
        other=None,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        alpha=1,
    ):
        assert input.dtype == weight.dtype, f"dtypes must match: {input.dtype}, {weight.dtype}"
        assert input.is_contiguous(), "input must be contiguous"
        assert weight.is_contiguous(), "weight must be contiguous"
        assert bias is None or bias.is_contiguous(), "bias must be contiguous"
        assert other is None or other.is_contiguous(), "other must be contiguous"

        assert input.dim() == 4, f"only accepts 4D tensors for now"
        assert weight.dim() == 4, f"only accepts 4D weight tensors for now"

        assert input.size(1) == weight.size(1) * groups, "in_chans must match"

        if bias is not None:
            assert bias.size(0) == weight.size(0), "bias and weight must match in out_chans"
        if other is not None:
            assert other.size(0) == weight.size(0), "other and weight must match in out_chans"

        N, C, H, W = input.shape
        out_channels, in_channels, K_H, K_W = weight.shape
        assert stride in [1, (1, 1)], "only stride=(1,1) is currently supported"
        assert dilation in [1, (1, 1)], "only dilation=(1,1) is currently supported"
        assert groups in [1, C], "grouping not yet supported"

        PADDED_INPUT = list(input.shape)
        PADDED_INPUT[2] += K_H - 1
        PADDED_INPUT[3] += K_W - 1
        PADDED_INPUT = tuple(PADDED_INPUT)

        PADDED_HEIGHT, PADDED_WIDTH = PADDED_INPUT[-2:]

        o = torch.empty(N, out_channels, PADDED_HEIGHT, PADDED_WIDTH, dtype=input.dtype, device=input.device)

        TILE_HEIGHT = 16
        TILE_WIDTH = 16

        grid = lambda meta: (
            triton.cdiv(N * out_channels, meta["OCS"]),
            triton.cdiv(PADDED_HEIGHT, TILE_HEIGHT),
            triton.cdiv(PADDED_WIDTH, TILE_WIDTH),
        )

        num_stages = 1
        num_warps = 8

        BLOCK_SIZE = 12

        P = K_H - 1
        Q = K_W - 1

        if isinstance(padding, str):
            if padding == "valid":
                padding = (0, 0)
            elif padding == "same":
                padding = (
                    (-(P * H + Q * W) // 2 + P * W // 2 + W) // W,
                    (-(P * H + Q * W) // 2 + Q * H // 2 + H) // H,
                )
            else:
                raise Exception("Unknown padding type")

        input = F.pad(input, (P, Q, P, Q))

        with torch.cuda.device(input.device.index):
            conv2d_forward[grid](
                input,
                weight,
                bias,
                other,
                o,
                *input.stride(),
                *weight.stride(),
                *o.stride(),
                alpha,
                PADDED_HEIGHT,
                PADDED_WIDTH,
                K_H,
                K_W,
                groups,
                H,
                W,
                *o.shape,
                BLOCK_SIZE=BLOCK_SIZE,
                TILE_HEIGHT=TILE_HEIGHT,
                TILE_WIDTH=TILE_WIDTH,
                num_warps=num_warps,
                num_stages=num_stages,
            )
        return o


def conv2d_add(input, weight, bias=None, other=None, stride=1, padding=0, dilation=1, groups=1, alpha=1):
    return Conv2dAdd.apply(input, weight, bias, other, stride, padding, dilation, groups, alpha)
