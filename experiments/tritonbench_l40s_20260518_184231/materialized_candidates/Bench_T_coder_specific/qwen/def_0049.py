import torch
import triton
import triton.language as tl

# Define the grid and block sizes for the convolution kernel
BLOCK_SIZE_N = 4
BLOCK_SIZE_C = 4
BLOCK_SIZE_H = 8
BLOCK_SIZE_W = 8

# Define the grid and block sizes for the Leaky ReLU kernel
BLOCK_SIZE_N_LEAKY = 4
BLOCK_SIZE_OC_LEAKY = 4
BLOCK_SIZE_H_LEAKY = 8
BLOCK_SIZE_W_LEAKY = 8

def leaky_relu_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, negative_slope=0.01, inplace=False) -> torch.Tensor:
    # Get the shape of the input tensor
    N, C, H, W = input.shape
    
    # Calculate the output shape
    OC = weight.shape[0]
    OH = (H + 2 * padding - dilation * (KH - 1) - 1) // stride + 1
    OW = (W + 2 * padding - dilation * (KW - 1) - 1) // stride + 1
    
    # Allocate memory for the output tensor
    output = torch.zeros((N, OC, OH, OW), device=input.device, dtype=input.dtype)
    
    # Run the convolution kernel
    num_warps = 4
    grid = (
        (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N,
        (OC // BLOCK_SIZE_C + 1) // BLOCK_SIZE_C,
        (OH + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H,
        (OW + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W,
    )
    conv2d_kernel[grid, (BLOCK_SIZE_N, BLOCK_SIZE_C, BLOCK_SIZE_H, BLOCK_SIZE_W), num_warps](
        input.contiguous(), weight.contiguous(), bias.contiguous() if bias is not None else None, output.contiguous(),
        N, C, H, W, OC, KH, KW, stride, stride, padding, padding, dilation, dilation, groups, stride, stride, padding, padding,
        BLOCK_SIZE_N, BLOCK_SIZE_C, BLOCK_SIZE_H, BLOCK_SIZE_W, groups
    )

    # Run the Leaky ReLU kernel
    grid_leaky = (
        (N + BLOCK_SIZE_N_LEAKY - 1) // BLOCK_SIZE_N_LEAKY,
        (OC // BLOCK_SIZE_OC_LEAKY + 1) // BLOCK_SIZE_OC_LEAKY,
        (OH + BLOCK_SIZE_H_LEAKY - 1) // BLOCK_SIZE_H_LEAKY,
        (OW + BLOCK_SIZE_W_LEAKY - 1) // BLOCK_SIZE_W_LEAKY,
    )
    leaky_relu_kernel[grid_leaky, (BLOCK_SIZE_N_LEAKY, BLOCK_SIZE_OC_LEAKY, BLOCK_SIZE_H_LEAKY, BLOCK_SIZE_W_LEAKY), num_warps](
        output.contiguous(), output.contiguous(),
        N, OC, OH, OW, negative_slope,
        BLOCK_SIZE_N_LEAKY, BLOCK_SIZE_OC_LEAKY, BLOCK_SIZE_H_LEAKY, BLOCK_SIZE_W_LEAKY
    )

    return output
