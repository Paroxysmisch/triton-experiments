import torch
import triton
import triton.language as tl

@triton.jit
def pixel_shuffle_kernel(
    input_ptr,
    output_ptr,
    input_n, input_c, input_h, input_w,
    output_n, output_c, output_h, output_w,
    upscale_factor,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < (output_n * output_c * output_h * output_w)

    # Convert flat offset to 4D indices (n, c, h, w) for output
    n_idx = offset // (output_c * output_h * output_w)
    remainder = offset % (output_c * output_h * output_w)
    c_idx = remainder // (output_h * output_w)
    remainder = remainder % (output_h * output_w)
    h_idx = remainder // output_w
    w_idx = remainder % output_w

    # Compute input indices
    r = upscale_factor
    h_input = h_idx // r
    w_input = w_idx // r
    sub_h = h_idx % r
    sub_w = w_idx % r
    c_input = c_idx * (r * r) + sub_h * r + sub_w

    # Input and output offsets calculation
    input_offset = (
        n_idx * input_c * input_h * input_w +
        c_input * input_h * input_w +
        h_input * input_w +
        w_input
    )
    output_offset = (
        n_idx * output_c * output_h * output_w +
        c_idx * output_h * output_w +
        h_idx * output_w +
        w_idx
    )

    input_val = tl.load(input_ptr + input_offset, mask=mask, other=0.0)
    tl.store(output_ptr + output_offset, input_val, mask=mask)


def pixel_shuffle_conv2d(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias=None,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    upscale_factor=2
) -> torch.Tensor:
    # Apply 2D convolution
    conv_out = torch.nn.functional.conv2d(
        input, weight, bias, stride, padding, dilation, groups
    ).contiguous()

    # Verify channels compatibility for pixel shuffle
    C = conv_out.size(1)
    r = upscale_factor
    if C % (r ** 2) != 0:
        raise ValueError(f"Channels {C} must be divisible by {r}^2 = {r**2} for pixel shuffle")

    # Determine output dimensions
    N, _, H, W = conv_out.shape
    output_c = C // (r ** 2)
    output_h = H * r
    output_w = W * r

    # Allocate output tensor
    output = torch.empty((N, output_c, output_h, output_w), device=conv_out.device, dtype=conv_out.dtype)

    # Launch Triton kernel for pixel shuffle
    total_elements = N * output_c * output_h * output_w
    BLOCK_SIZE = 1024  # Tune this based on hardware specifics
    grid = lambda meta: (triton.cdiv(total_elements, meta['BLOCK_SIZE']), )

    pixel_shuffle_kernel[grid](
        conv_out,
        output,
        N, C, H, W,
        N, output_c, output_h, output_w,
        upscale_factor,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output
