import torch
import triton
import triton.language as tl
from torch import Tensor

@triton.jit
def conv2d_kernel(x, conv_weight, conv_bias, output, stride, padding, n_rows, n_cols, n_channels, n_output_channels):
    row_id = tl.program_id(0)
    col_id = tl.program_id(1)
    
    # Calculate the output position
    out_row = row_id * stride
    out_col = col_id * stride

    # Load weights and bias
    weight = tl.load(conv_weight)
    bias = tl.load(conv_bias) if conv_bias is not None else 0.0

    # Initialize the output
    acc = bias

    # Perform convolution
    for k in range(n_channels):
        for i in range(-padding, 3 - padding):
            for j in range(-padding, 3 - padding):
                if 0 <= out_row + i < n_rows and 0 <= out_col + j < n_cols:
                    acc += tl.load(x + (out_row + i) * n_cols + (out_col + j) * n_channels + k) * weight[k, i + padding, j + padding]

    tl.store(output + row_id * n_cols + col_id, acc)

@triton.jit
def layer_norm_kernel(x, weight, output, n_elements, eps):
    row_id = tl.program_id(0)
    offsets = tl.arange(0, n_elements)

    # Load input
    x_row = x + row_id * n_elements
    x_vals = tl.load(x_row + offsets)

    # Compute mean and variance
    mean = tl.sum(x_vals) / n_elements
    var = tl.sum((x_vals - mean) ** 2) / n_elements

    # Normalize
    norm_x = (x_vals - mean) / tl.sqrt(var + eps)

    # Scale with weight
    output_vals = norm_x * tl.load(weight)
    tl.store(output + row_id * n_elements + offsets, output_vals)

@triton.jit
def silu_kernel(x, output, n_elements):
    row_id = tl.program_id(0)
    offsets = tl.arange(0, n_elements)

    # Load input
    x_row = x + row_id * n_elements
    x_vals = tl.load(x_row + offsets)

    # Apply SiLU activation
    output_vals = x_vals * torch.sigmoid(x_vals)
    tl.store(output + row_id * n_elements + offsets, output_vals)

@torch.inference_mode()
def fused_silu_layer_norm_conv2d(x: Tensor, weight: Tensor, conv_weight: Tensor, conv_bias: Tensor = None, conv_stride: int = 1, conv_padding: int = 0, conv_dilation: int = 1, conv_groups: int = 1, ln_eps: float = 1e-5) -> Tensor:
    n_output_channels = conv_weight.shape[0]
    n_channels = conv_weight.shape[1]
    n_rows, n_cols = x.shape[2], x.shape[3]

    # Prepare output tensors
    conv_out = torch.empty((x.shape[0], n_output_channels, n_rows, n_cols), device=x.device)
    norm_out = torch.empty_like(conv_out)

    # Launch convolution kernel
    grid = (n_rows, n_cols)
    conv2d_kernel[grid](x, conv_weight, conv_bias, conv_out, conv_stride, conv_padding, n_rows, n_cols, n_channels, n_output_channels)

    # Launch layer normalization kernel
    layer_norm_kernel[grid](conv_out, weight, norm_out, n_output_channels, ln_eps)

    # Launch SiLU activation kernel
    output = torch.empty_like(norm_out)
    silu_kernel[grid](norm_out, output, n_output_channels)

    return output
