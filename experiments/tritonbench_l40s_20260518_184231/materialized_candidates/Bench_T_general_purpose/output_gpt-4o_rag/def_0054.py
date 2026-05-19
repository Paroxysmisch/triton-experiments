import torch
import triton
import triton.language as tl

@triton.jit
def conv2d_gelu_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    N, C, H, W, K, R, S, stride_h, stride_w,
    pad_h, pad_w, dilation_h, dilation_w,
    out_h, out_w, groups, approximate, BLOCK_SIZE: tl.constexpr
):
    # Calculate the batch, output channel, and spatial position
    batch_id = tl.program_id(0)
    out_channel_id = tl.program_id(1)
    out_y = tl.program_id(2) // out_w
    out_x = tl.program_id(2) % out_w

    # Initialize output
    result = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    # Calculate input positions
    in_y = out_y * stride_h - pad_h
    in_x = out_x * stride_w - pad_w

    # Perform convolution
    for c in range(C // groups):
        for r in range(R):
            for s in range(S):
                in_y_pos = in_y + r * dilation_h
                in_x_pos = in_x + s * dilation_w
                if 0 <= in_y_pos < H and 0 <= in_x_pos < W:
                    input_val = tl.load(input_ptr + ((batch_id * C + c) * H + in_y_pos) * W + in_x_pos)
                    weight_val = tl.load(weight_ptr + ((out_channel_id * (C // groups) + c) * R + r) * S + s)
                    result += input_val * weight_val

    # Add bias if present
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + out_channel_id)
        result += bias_val

    # Apply GELU
    if approximate == 0:  # 'none'
        gelu_result = result * 0.5 * (1.0 + tl.erf(result / tl.sqrt(2.0)))
    else:  # 'tanh'
        gelu_result = 0.5 * result * (1.0 + tl.tanh(tl.sqrt(2.0 / 3.141592653589793) * (result + 0.044715 * result**3)))

    # Store result
    tl.store(output_ptr + ((batch_id * K + out_channel_id) * out_h + out_y) * out_w + out_x, gelu_result)

def gelu_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, approximate='none', out=None):
    # Ensure input and weight are on CUDA
    assert input.device.type == 'cuda', "Input tensor must be on a CUDA device."
    assert weight.device == input.device, "Weight tensor must be on the same CUDA device as input."

    # Get input dimensions
    N, C, H, W = input.shape
    K, _, R, S = weight.shape

    # Calculate output dimensions
    stride_h, stride_w = (stride, stride) if isinstance(stride, int) else stride
    pad_h, pad_w = (padding, padding) if isinstance(padding, int) else padding
    dilation_h, dilation_w = (dilation, dilation) if isinstance(dilation, int) else dilation

    out_h = (H + 2 * pad_h - dilation_h * (R - 1) - 1) // stride_h + 1
    out_w = (W + 2 * pad_w - dilation_w * (S - 1) - 1) // stride_w + 1

    # Prepare output tensor
    if out is None:
        out = torch.empty((N, K, out_h, out_w), device=input.device, dtype=input.dtype)

    # Convert approximate to integer for kernel
    approximate_flag = 0 if approximate == 'none' else 1

    # Define grid
    BLOCK_SIZE = 256
    grid = (N, K, out_h * out_w)

    # Launch Triton kernel
    conv2d_gelu_kernel[grid](
        input_ptr=input, weight_ptr=weight, bias_ptr=bias, output_ptr=out,
        N=N, C=C, H=H, W=W, K=K, R=R, S=S,
        stride_h=stride_h, stride_w=stride_w,
        pad_h=pad_h, pad_w=pad_w,
        dilation_h=dilation_h, dilation_w=dilation_w,
        out_h=out_h, out_w=out_w, groups=groups, approximate=approximate_flag,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return out
