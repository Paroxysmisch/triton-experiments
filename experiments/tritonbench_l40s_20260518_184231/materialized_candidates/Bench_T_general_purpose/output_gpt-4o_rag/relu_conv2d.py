import torch
import triton
import triton.language as tl

@triton.jit
def conv2d_relu_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    B, H, W, C_in, C_out, K_H, K_W,
    stride_h, stride_w, pad_h, pad_w,
    BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_W: tl.constexpr
):
    # Calculate the position of the current block in the output tensor
    b_idx = tl.program_id(0)
    h_idx = tl.program_id(1) * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    w_idx = tl.program_id(2) * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)

    # Compute the convolution
    output = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)

    for c_in in range(C_in):
        for k_h in range(K_H):
            for k_w in range(K_W):
                # Calculate input positions
                in_h = h_idx * stride_h + k_h - pad_h
                in_w = w_idx * stride_w + k_w - pad_w

                # Load input values with boundary checks
                mask_h = (in_h >= 0) & (in_h < H)
                mask_w = (in_w >= 0) & (in_w < W)
                mask = mask_h[:, None] & mask_w[None, :]

                input_vals = tl.load(input_ptr + ((b_idx * C_in + c_in) * H + in_h)[:, None] * W + in_w[None, :], mask=mask, other=0.0)

                # Load weight values
                weight_vals = tl.load(weight_ptr + (c_in * K_H + k_h) * K_W + k_w)

                # Accumulate convolution results
                output += input_vals * weight_vals

    # Apply bias if provided
    if bias_ptr:
        bias = tl.load(bias_ptr + tl.arange(0, C_out))
        output += bias

    # Apply ReLU activation
    output = tl.maximum(output, 0.0)

    # Store the result
    tl.store(output_ptr + ((b_idx * C_out) * H + h_idx)[:, None] * W + w_idx[None, :], output)

def relu_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, inplace=False):
    # Ensure input and weight are on CUDA
    device = input.device
    assert device.type == "cuda", "Input tensor must be on a CUDA device for Triton ops."
    assert input.device == weight.device, "Input and weight must be on the same CUDA device."

    # Get input dimensions
    B, C_in, H, W = input.shape
    C_out, _, K_H, K_W = weight.shape

    # Calculate output dimensions
    stride_h, stride_w = (stride, stride) if isinstance(stride, int) else stride
    pad_h, pad_w = (padding, padding) if isinstance(padding, int) else padding
    out_H = (H + 2 * pad_h - K_H) // stride_h + 1
    out_W = (W + 2 * pad_w - K_W) // stride_w + 1

    # Allocate output tensor
    output = torch.empty((B, C_out, out_H, out_W), device=device, dtype=input.dtype)

    # Launch Triton kernel
    BLOCK_SIZE_H = 16
    BLOCK_SIZE_W = 16
    grid = (B, (out_H + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H, (out_W + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W)

    conv2d_relu_kernel[grid](
        input_ptr=input,
        weight_ptr=weight,
        bias_ptr=bias if bias is not None else tl.nullptr,
        output_ptr=output,
        B=B, H=H, W=W, C_in=C_in, C_out=C_out, K_H=K_H, K_W=K_W,
        stride_h=stride_h, stride_w=stride_w, pad_h=pad_h, pad_w=pad_w,
        BLOCK_SIZE_H=BLOCK_SIZE_H, BLOCK_SIZE_W=BLOCK_SIZE_W
    )

    return output
