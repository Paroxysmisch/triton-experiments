import torch
import triton
import triton.language as tl

@triton.jit
def relu_conv2d_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    in_channels, out_channels, iH, iW, kH, kW,
    stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w,
    groups, inplace, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid = tl.program_id(0)
    # Calculate the output dimensions
    oH = (iH + 2 * padding_h - dilation_h * (kH - 1) - 1) // stride_h + 1
    oW = (iW + 2 * padding_w - dilation_w * (kW - 1) - 1) // stride_w + 1

    # Compute starting point for this block
    oh = pid // oW
    ow = pid % oW

    # Initialize accumulators for output
    output_val = 0.0

    # Iterate over each group
    for g in range(groups):
        # Iterate over each output channel
        for oc in range(out_channels // groups):
            # Initialize accumulator for this output channel
            acc = 0.0
            # Iterate over each input channel
            for ic in range(in_channels // groups):
                # Iterate over each kernel element
                for kh in range(kH):
                    for kw in range(kW):
                        ih = oh * stride_h + kh * dilation_h - padding_h
                        iw = ow * stride_w + kw * dilation_w - padding_w
                        if 0 <= ih < iH and 0 <= iw < iW:
                            input_idx = ((g * (in_channels // groups) + ic) * iH + ih) * iW + iw
                            weight_idx = (((g * (out_channels // groups) + oc) * (in_channels // groups) + ic) * kH + kh) * kW + kw
                            acc += tl.load(input_ptr + input_idx) * tl.load(weight_ptr + weight_idx)
            # Add bias if present
            if bias_ptr is not None:
                acc += tl.load(bias_ptr + g * (out_channels // groups) + oc)
            # Apply ReLU
            acc = tl.max(acc, 0.0)
            # Store the result
            output_idx = ((g * (out_channels // groups) + oc) * oH + oh) * oW + ow
            if inplace:
                tl.store(input_ptr + output_idx, acc)
            else:
                tl.store(output_ptr + output_idx, acc)

def relu_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, inplace=False):
    # Ensure stride, padding, and dilation are tuples
    stride = (stride, stride) if isinstance(stride, int) else stride
    padding = (padding, padding) if isinstance(padding, int) else padding
    dilation = (dilation, dilation) if isinstance(dilation, int) else dilation

    # Extract dimensions
    minibatch, in_channels, iH, iW = input.shape
    out_channels, _, kH, kW = weight.shape

    # Calculate output dimensions
    oH = (iH + 2 * padding[0] - dilation[0] * (kH - 1) - 1) // stride[0] + 1
    oW = (iW + 2 * padding[1] - dilation[1] * (kW - 1) - 1) // stride[1] + 1

    # Allocate output tensor
    if inplace:
        output = input
    else:
        output = torch.empty((minibatch, out_channels, oH, oW), device=input.device, dtype=input.dtype)

    # Launch Triton kernel
    grid = (oH * oW,)
    relu_conv2d_kernel[grid](
        input_ptr=input,
        weight_ptr=weight,
        bias_ptr=bias,
        output_ptr=output,
        in_channels=in_channels,
        out_channels=out_channels,
        iH=iH,
        iW=iW,
        kH=kH,
        kW=kW,
        stride_h=stride[0],
        stride_w=stride[1],
        padding_h=padding[0],
        padding_w=padding[1],
        dilation_h=dilation[0],
        dilation_w=dilation[1],
        groups=groups,
        inplace=inplace,
        BLOCK_M=32,  # Example block size, adjust as needed
        BLOCK_N=32   # Example block size, adjust as needed
    )

    return output
