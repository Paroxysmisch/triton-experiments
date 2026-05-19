import triton
import triton.language as tl

@triton.jit
def conv2d_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    input_shape, weight_shape, output_shape,
    stride, padding, dilation, groups,
    upscale_factor,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    # Extract shapes
    N, C, H, W = input_shape
    K, Cg, R, S = weight_shape
    Ng, Cg, Hg, Wg = output_shape
    G = groups
    stride_h, stride_w = stride
    padding_h, padding_w = padding
    dilation_h, dilation_w = dilation

    # Compute output dimensions
    Ho = (H + 2 * padding_h - dilation_h * (R - 1) - 1) // stride_h + 1
    Wo = (W + 2 * padding_w - dilation_w * (S - 1) - 1) // stride_w + 1

    # Compute pixel shuffle dimensions
    Ho_out = Ho * upscale_factor
    Wo_out = Wo * upscale_factor

    # Compute the block indices
    pid_n = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2)
    pid_w = tl.program_id(axis=3)

    # Compute the block ranges
    n_start = pid_n * BLOCK_SIZE_N
    k_start = pid_k * BLOCK_SIZE_K
    h_start = pid_h * BLOCK_SIZE_H
    w_start = pid_w * BLOCK_SIZE_W

    # Initialize output block
    output_block = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_K, BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)

    # Load input and weight blocks
    for r in range(R):
        for s in range(S):
            for g in range(G):
                input_block = tl.load(input_ptr + (n_start, g * Cg + c, h_start + r * dilation_h - padding_h, w_start + s * dilation_w - padding_w))
                weight_block = tl.load(weight_ptr + (k_start, g * Cg + c, r, s))
                output_block += tl.dot(input_block, weight_block)

    # Add bias if provided
    if bias_ptr is not None:
        bias_block = tl.load(bias_ptr + k_start)
        output_block += bias_block

    # Apply pixel shuffle
    for n in range(BLOCK_SIZE_N):
        for k in range(BLOCK_SIZE_K):
            for h in range(BLOCK_SIZE_H):
                for w in range(BLOCK_SIZE_W):
                    h_out = h * upscale_factor + (k // (K // (upscale_factor * upscale_factor)))
                    w_out = w * upscale_factor + (k % (upscale_factor * upscale_factor))
                    tl.store(output_ptr + (n_start + n, k_start + k, h_out, w_out), output_block[n, k, h, w])

# Define the wrapper function
def pixel_shuffle_conv2d(input: torch.Tensor, weight: torch.Tensor, bias=None, stride=1, padding=0, dilation=1, groups=1, upscale_factor=2) -> torch.Tensor:
    # Ensure input and weight are contiguous
    input = input.contiguous()
    weight = weight.contiguous()

    # Extract shapes
    N, C, H, W = input.shape
    K, Cg, R, S = weight.shape
    G = groups

    # Compute output dimensions
    Ho = (H + 2 * padding - dilation * (R - 1) - 1) // stride + 1
    Wo = (W + 2 * padding - dilation * (S - 1) - 1) // stride + 1

    # Compute pixel shuffle dimensions
    Ho_out = Ho * upscale_factor
    Wo_out = Wo * upscale_factor

    # Allocate output tensor
    output = torch.empty((N, K, Ho_out, Wo_out), device=input.device, dtype=input.dtype)

    # Define grid and block sizes
    grid = (N, K, Ho, Wo)
    block = (16, 16, 1, 1)

    # Launch the kernel
    conv2d_kernel[grid](
        input, weight, bias, output,
        (N, C, H, W), (K, Cg, R, S), (N, K, Ho, Wo),
        (stride, stride), (padding, padding), (dilation, dilation), groups,
        upscale_factor,
        *block
    )

    return output

import torch

# Example input and weight tensors
input = torch.randn(1, 4, 8, 8, device='cuda')
weight = torch.randn(16, 4, 3, 3, device='cuda')
bias = torch.randn(16, device='cuda')

# Call the wrapper function
output = pixel_shuffle_conv2d(input, weight, bias, stride=1, padding=1, dilation=1, groups=1, upscale_factor=2)

# Print the output shape
print(output.shape)  # Expected shape: (1, 16, 16, 16)
