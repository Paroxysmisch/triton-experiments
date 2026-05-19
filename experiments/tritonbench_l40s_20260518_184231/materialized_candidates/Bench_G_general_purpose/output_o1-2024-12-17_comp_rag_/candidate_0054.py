import triton
import triton.language as tl
import torch

@triton.jit
def conv2d_forward_kernel(
    input_ptr,                   # *Pointer to input data
    weight_ptr,                  # *Pointer to weights
    output_ptr,                  # *Pointer to output
    batch_size, in_channels, in_height, in_width,
    out_channels, out_height, out_width,
    kernel_h, kernel_w,
    stride_h, stride_w,
    pad_h, pad_w,
    groups,
    input_batch_stride, input_channel_stride, input_height_stride, input_width_stride,
    weight_out_channel_stride, weight_in_channel_stride,
    weight_kernel_h_stride, weight_kernel_w_stride,
    output_batch_stride, output_channel_stride, output_height_stride, output_width_stride,
    ACC_TYPE: tl.constexpr,      # Accumulator type (e.g. tl.float32)
    ALLOW_FP16: tl.constexpr,    # Whether FP16 is allowed
    ALLOW_TF32: tl.constexpr,    # Whether TF32 is allowed
    BLOCK_BATCH: tl.constexpr,   # Block size for batch dimension
    BLOCK_OUT_CHANNEL: tl.constexpr,   # Block size for out_channels dimension
    BLOCK_OUT_HEIGHT: tl.constexpr,    # Block size for out_height dimension
    BLOCK_OUT_WIDTH: tl.constexpr      # Block size for out_width dimension
):
    # Current program (thread block) identifiers
    pid_b = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_ohw = tl.program_id(2)

    # Offsets within each dimension for this program
    b_offset = pid_b * BLOCK_BATCH
    oc_offset = pid_oc * BLOCK_OUT_CHANNEL
    oh_offset = (pid_ohw // (out_width // BLOCK_OUT_WIDTH + (1 if out_width % BLOCK_OUT_WIDTH != 0 else 0))) * BLOCK_OUT_HEIGHT
    ow_offset = (pid_ohw % (out_width // BLOCK_OUT_WIDTH + (1 if out_width % BLOCK_OUT_WIDTH != 0 else 0))) * BLOCK_OUT_WIDTH

    # Declare result tile
    output_acc = tl.zeros((BLOCK_BATCH, BLOCK_OUT_CHANNEL, BLOCK_OUT_HEIGHT, BLOCK_OUT_WIDTH), dtype=ACC_TYPE)

    # Loop over group and in_channels
    group_size = in_channels // groups
    # For each output channel in the block
    for oc_inner in range(BLOCK_OUT_CHANNEL):
        oc = oc_offset + oc_inner
        if oc >= out_channels:
            break

    # For each point in the output tile, compute the convolution
    for oh_inner in range(BLOCK_OUT_HEIGHT):
        oh = oh_offset + oh_inner
        if oh >= out_height:
            break
        for ow_inner in range(BLOCK_OUT_WIDTH):
            ow = ow_offset + ow_inner
            if ow >= out_width:
                break

            for group_idx in range(groups):
                # Check if oc belongs to this group
                if (oc_offset // (out_channels // groups)) != group_idx and groups > 1:
                    continue

                # Within the group, handle all input channels
                for ic in range(group_size):
                    ic_global = group_idx * group_size + ic
                    if ic_global >= in_channels:
                        break

                    # Compute the start in the input image
                    ih_start = oh * stride_h - pad_h
                    iw_start = ow * stride_w - pad_w

                    # For each kernel element
                    for kh in range(kernel_h):
                        ih = ih_start + kh
                        if ih < 0 or ih >= in_height:
                            continue
                        for kw in range(kernel_w):
                            iw = iw_start + kw
                            if iw < 0 or iw >= in_width:
                                continue

                            # For each batch in the block
                            for bb in range(BLOCK_BATCH):
                                b = b_offset + bb
                                if b >= batch_size:
                                    break

                                # Load input
                                inp_val = tl.load(
                                    input_ptr
                                    + b * input_batch_stride
                                    + ic_global * input_channel_stride
                                    + ih * input_height_stride
                                    + iw * input_width_stride,
                                    mask=(b < batch_size and ic_global < in_channels and
                                          0 <= ih < in_height and 0 <= iw < in_width)
                                )
                                # Load weight
                                w_val = tl.load(
                                    weight_ptr
                                    + oc * weight_out_channel_stride
                                    + ic_global * weight_in_channel_stride
                                    + kh * weight_kernel_h_stride
                                    + kw * weight_kernel_w_stride,
                                    mask=(oc < out_channels and ic_global < in_channels
                                          and kh < kernel_h and kw < kernel_w)
                                )
                                output_acc[bb, oc_inner, oh_inner, ow_inner] += inp_val * w_val

    # Store results
    for bb in range(BLOCK_BATCH):
        b = b_offset + bb
        if b >= batch_size:
            break
        for oc_inner in range(BLOCK_OUT_CHANNEL):
            oc = oc_offset + oc_inner
            if oc >= out_channels:
                break
            for oh_inner in range(BLOCK_OUT_HEIGHT):
                oh = oh_offset + oh_inner
                if oh >= out_height:
                    break
                for ow_inner in range(BLOCK_OUT_WIDTH):
                    ow = ow_offset + ow_inner
                    if ow >= out_width:
                        break
                    val = output_acc[bb, oc_inner, oh_inner, ow_inner]
                    tl.store(
                        output_ptr
                        + b * output_batch_stride
                        + oc * output_channel_stride
                        + oh * output_height_stride
                        + ow * output_width_stride,
                        val,
                        mask=(b < batch_size and oc < out_channels and oh < out_height and ow < out_width)
                    )

def conv2d_forward(
    input: torch.Tensor,
    weight: torch.Tensor,
    kernel_size: tuple,
    stride: tuple,
    padding: tuple,
    groups: int = 1,
    allow_fp16: bool = True,
    allow_tf32: bool = False,
):
    batch_size, in_channels, in_height, in_width = input.shape
    out_channels, _, kernel_h, kernel_w = weight.shape

    # Compute output height and width
    out_height = (in_height + 2 * padding[0] - kernel_h) // stride[0] + 1
    out_width = (in_width + 2 * padding[1] - kernel_w) // stride[1] + 1

    # Prepare output
    output = torch.empty(
        (batch_size, out_channels, out_height, out_width),
        dtype=input.dtype,
        device=input.device
    )

    # Strides for input, weight, output
    input_batch_stride = in_channels * in_height * in_width
    input_channel_stride = in_height * in_width
    input_height_stride = in_width
    input_width_stride = 1

    weight_out_channel_stride = weight.shape[1] * kernel_h * kernel_w
    weight_in_channel_stride = kernel_h * kernel_w
    weight_kernel_h_stride = kernel_w
    weight_kernel_w_stride = 1

    output_batch_stride = out_channels * out_height * out_width
    output_channel_stride = out_height * out_width
    output_height_stride = out_width
    output_width_stride = 1

    # Block and grid
    BLOCK_BATCH = 1
    BLOCK_OUT_CHANNEL = 16
    BLOCK_OUT_HEIGHT = 8
    BLOCK_OUT_WIDTH = 8

    grid_b = (batch_size + BLOCK_BATCH - 1) // BLOCK_BATCH
    grid_oc = (out_channels + BLOCK_OUT_CHANNEL - 1) // BLOCK_OUT_CHANNEL
    grid_ohw = ((out_height + BLOCK_OUT_HEIGHT - 1) // BLOCK_OUT_HEIGHT) * (
        (out_width + BLOCK_OUT_WIDTH - 1) // BLOCK_OUT_WIDTH
    )

    conv2d_forward_kernel[grid_b, grid_oc, grid_ohw](
        input_ptr=input,
        weight_ptr=weight,
        output_ptr=output,
        batch_size=batch_size,
        in_channels=in_channels,
        in_height=in_height,
        in_width=in_width,
        out_channels=out_channels,
        out_height=out_height,
        out_width=out_width,
        kernel_h=kernel_h,
        kernel_w=kernel_w,
        stride_h=stride[0],
        stride_w=stride[1],
        pad_h=padding[0],
        pad_w=padding[1],
        groups=groups,
        input_batch_stride=input_batch_stride,
        input_channel_stride=input_channel_stride,
        input_height_stride=input_height_stride,
        input_width_stride=input_width_stride,
        weight_out_channel_stride=weight_out_channel_stride,
        weight_in_channel_stride=weight_in_channel_stride,
        weight_kernel_h_stride=weight_kernel_h_stride,
        weight_kernel_w_stride=weight_kernel_w_stride,
        output_batch_stride=output_batch_stride,
        output_channel_stride=output_channel_stride,
        output_height_stride=output_height_stride,
        output_width_stride=output_width_stride,
        ACC_TYPE=tl.float32,
        ALLOW_FP16=allow_fp16,
        ALLOW_TF32=allow_tf32,
        BLOCK_BATCH=BLOCK_BATCH,
        BLOCK_OUT_CHANNEL=BLOCK_OUT_CHANNEL,
        BLOCK_OUT_HEIGHT=BLOCK_OUT_HEIGHT,
        BLOCK_OUT_WIDTH=BLOCK_OUT_WIDTH,
    )

    return output
