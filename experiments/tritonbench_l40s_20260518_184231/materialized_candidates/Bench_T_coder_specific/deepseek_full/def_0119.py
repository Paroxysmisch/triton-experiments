import torch
import triton
import triton.language as tl

@triton.jit
def pixel_shuffle_conv2d_kernel(
    input_pointer,
    output_pointer,
    weight_pointer,
    bias_pointer,
    input_batch,
    input_channel,
    input_height,
    input_width,
    output_channel,
    kernel_height,
    kernel_width,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    dilation_h,
    dilation_w,
    groups,
    upscale_factor,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    OUTPUT_BLOCK_SIZE_H: tl.constexpr,
    OUTPUT_BLOCK_SIZE_W: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    pim = tl.program_id(axis=1)

    num_pid_h = tl.cdiv(input_height, BLOCK_SIZE_H)
    num_pid_w = tl.cdiv(input_width, BLOCK_SIZE_W)
    num_pid_in_group = groups * num_pid_h * num_pid_w
    group_id = pid // num_pid_in_group
    first_pid_h = pid - group_id * num_pid_in_group
    first_pid_w = 0

    input_channel_per_group = input_channel // groups
    output_channel_per_group = output_channel // groups

    input_pointer += (
        pim // input_channel_per_group * input_height * input_width
        + (pim % input_channel_per_group // input_width) * input_height
        + tl.arange(0, BLOCK_SIZE_H)
    )
    input_pointer += (
        (pim % input_channel_per_group % input_width // BLOCK_SIZE_W) * input_width
        + tl.arange(0, BLOCK_SIZE_W)
    )
    weight_pointer += (
        output_channel_per_group * input_channel_per_group * kernel_height * kernel_width
        + (pim % input_channel_per_group // input_width) * kernel_height
        + tl.arange(0, BLOCK_SIZE_H)
    )
    weight_pointer += (
        (pim % input_channel_per_group % input_width // BLOCK_SIZE_W) * kernel_width
        + tl.arange(0, BLOCK_SIZE_W)
    )

    shuffle_weight_pointer = weight_pointer
    shuffle_weight_stride_h = (
        output_channel_per_group * kernel_height * kernel_width
    ) // upscale_factor
    shuffle_weight_stride_w = (
        output_channel_per_group * kernel_height * kernel_width
    ) // (upscale_factor * upscale_factor)

    bias_pointer += pim % output_channel_per_group

    output_pointer += (
        group_id * input_batch * output_channel_per_group * (input_height * upscale_factor) * (input_width * upscale_factor)
        + (pim % output_channel_per_group // (output_channel_per_group // input_height)) * (input_height * upscale_factor)
        + tl.arange(0, OUTPUT_BLOCK_SIZE_H)
    )
    output_pointer += (
        (pim % output_channel_per_group % (output_channel_per_group // input_height) // (input_height // OUTPUT_BLOCK_SIZE_H))
        * (input_height * upscale_factor)
        + tl.arange(0, OUTPUT_BLOCK_SIZE_W)
    )

    shuffle_output_pointer = output_pointer
    shuffle_output_stride_h = output_channel_per_group * (input_height * upscale_factor) * (input_width * upscale_factor)
    shuffle_output_stride_w = (input_height * upscale_factor) * (input_width * upscale_factor)

    offs_h = (first_pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)) % input_height
    offs_w = (first_pid_w * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)) % input_width

    shuffle_offs_h = tl.arange(0, OUTPUT_BLOCK_SIZE_H)
    shuffle_offs_w = tl.arange(0, OUTPUT_BLOCK_SIZE_W)

    input_mask = offs_h[:, None] < input_height and offs_w[None, :] < input_width

    shuffle_input_mask = shuffle_offs_h[:, None] < (input_height * upscale_factor) and shuffle_offs_w[None, :] < (
        input_width * upscale_factor
    )

    accumulator = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
    for c in range(0, input_channel_per_group):
        for kh in range(0, kernel_height):
            for kw in range(0, kernel_width):
                input_idx = input_pointer + c * input_height * input_width + (offs_h + kh * dilation_h - pad_h) * input_width + (
                    offs_w + kw * dilation_w - pad_w
                )
                weight_idx = weight_pointer + c * kernel_height * kernel_width + kh * kernel_width + kw
                bias_idx = bias_pointer + c

                shuffle_weight_idx = (
                    shuffle_weight_pointer
                    + c * kernel_height * kernel_width
                    + (kh // upscale_factor) * kernel_height * kernel_width * upscale_factor
                    + (kw // upscale_factor) * upscale_factor
                )
                shuffle_bias_idx = bias_pointer + c

                shuffle_input_pointer = input_idx
                for shuffle_h in range(0, upscale_factor):
                    for shuffle_w in range(0, upscale_factor):
                        shuffle_input_idx = (
                            shuffle_input_pointer
                            + (shuffle_offs_h + shuffle_h) * (input_width * upscale_factor)
                            + (shuffle_offs_w + shuffle_w)
                        )

                        shuffle_output_idx = (
                            shuffle_output_pointer
                            + shuffle_h * shuffle_output_stride_h
                            + shuffle_w * shuffle_output_stride_w
                        )

                        shuffle_input = tl.load(
                            shuffle_input_idx, mask=shuffle_input_mask, other=0
                        )
                        shuffle_weight = tl.load(
                            shuffle_weight_idx, mask=shuffle_input_mask, other=0
                        )
                        shuffle_bias = tl.load(shuffle_bias_idx)

                        tl.store(
                            shuffle_output_idx,
                            shuffle_input + shuffle_weight * shuffle_bias,
                            mask=shuffle_input_mask,
                        )

                input_value = tl.load(input_idx, mask=input_mask, other=0)
                weight_value = tl.load(weight_idx, mask=input_mask, other=0)
                bias_value = tl.load(bias_idx)

                accumulator += input_value * weight_value

        accumulator += bias_value

    tl.store(output_pointer, accumulator, mask=input_mask)

def pixel_shuffle_conv2d(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    stride: int = 1,
    padding: int = 0,
