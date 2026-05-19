import torch
import triton
import triton.language as tl

@triton.jit
def conv2d_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    batch_size, in_channels, out_channels, iH, iW, kH, kW,
    stride, padding, dilation, BLOCK_SIZE: tl.constexpr
):
    # Triton kernel for 2D convolution
    batch_id = tl.program_id(0)
    out_channel_id = tl.program_id(1)
    h_idx = tl.program_id(2)
    w_idx = tl.program_id(3)

    # Calculate input dimensions
    oH = (iH + 2 * padding - dilation * (kH - 1) - 1) // stride + 1
    oW = (iW + 2 * padding - dilation * (kW - 1) - 1) // stride + 1

    # Initialize output
    result = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)

    # Loop over input channels and kernel size
    for c in range(in_channels):
        for kh in range(kH):
            for kw in range(kW):
                # Calculate input position
                ih = h_idx * stride + kh * dilation - padding
                iw = w_idx * stride + kw * dilation - padding
                # Check if within bounds
                if 0 <= ih < iH and 0 <= iw < iW:
                    input_val = tl.load(input_ptr + batch_id * in_channels * iH * iW + c * iH * iW + ih * iW + iw)
                    weight_val = tl.load(weight_ptr + out_channel_id * in_channels * kH * kW + c * kH * kW + kh * kW + kw)
                    result += input_val * weight_val

    # Add bias if present
    if bias_ptr:
        bias_val = tl.load(bias_ptr + out_channel_id)
        result += bias_val

    # Store the result
    tl.store(output_ptr + batch_id * out_channels * oH * oW + out_channel_id * oH * oW + h_idx * oW + w_idx, result)

def pixel_shuffle_conv2d(input: torch.Tensor, weight: torch.Tensor, bias=None, stride=1, padding=0, dilation=1, groups=1, upscale_factor=2) -> torch.Tensor:
    assert input.device.type == "cuda", "Input tensor must be on a CUDA device for Triton ops."
    assert input.device == weight.device, "Input and weight must be on the same CUDA device."

    # Input dimensions
    batch_size, in_channels, iH, iW = input.shape
    out_channels, _, kH, kW = weight.shape

    # Calculate output dimensions
    oH = (iH + 2 * padding - dilation * (kH - 1) - 1) // stride + 1
    oW = (iW + 2 * padding - dilation * (kW - 1) - 1) // stride + 1
    out_channels *= upscale_factor ** 2

    # Initialize output
    output = torch.empty((batch_size, out_channels, oH, oW), device=input.device, dtype=input.dtype)

    # Define grid and block size
    BLOCK_SIZE = 16
    grid = (batch_size, out_channels // (upscale_factor ** 2), oH, oW)

    # Launch Triton kernel
    conv2d_kernel[grid](
        input_ptr=input,
        weight_ptr=weight,
        bias_ptr=bias if bias is not None else tl.zeros(1, dtype=tl.float32),
        output_ptr=output,
        batch_size=batch_size, in_channels=in_channels, out_channels=out_channels // (upscale_factor ** 2),
        iH=iH, iW=iW, kH=kH, kW=kW,
        stride=stride, padding=padding, dilation=dilation,
        BLOCK_SIZE=BLOCK_SIZE
    )

    # Pixel shuffle operation
    output = output.view(batch_size, out_channels // (upscale_factor ** 2), upscale_factor, upscale_factor, oH, oW)
    output = output.permute(0, 1, 4, 2, 5, 3).contiguous()
    output = output.view(batch_size, out_channels, oH * upscale_factor, oW * upscale_factor)

    return output
