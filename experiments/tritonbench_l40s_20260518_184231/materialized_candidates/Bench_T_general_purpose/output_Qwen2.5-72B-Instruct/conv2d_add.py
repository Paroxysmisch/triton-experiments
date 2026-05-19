import triton
import triton.language as tl

@triton.jit
def conv2d_add_kernel(
    input_ptr, weight_ptr, bias_ptr, other_ptr, out_ptr,
    input_shape, weight_shape, bias_shape, other_shape, out_shape,
    stride, padding, dilation, groups, alpha,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_C: tl.constexpr, BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_W: tl.constexpr
):
    # Extract shapes and strides
    N, C, H, W = input_shape
    O, Cg, K_H, K_W = weight_shape
    B = bias_shape[0] if bias_ptr is not None else 0
    Oth = other_shape[0] if other_ptr is not None else 0
    _, _, OH, OW = out_shape

    # Compute padding and dilation
    pad_H, pad_W = padding
    dil_H, dil_W = dilation

    # Compute the output indices
    pid_n = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)
    pid_w = tl.program_id(axis=2)

    # Compute the input and output block indices
    n = pid_n * BLOCK_SIZE_N
    h = pid_h * BLOCK_SIZE_H
    w = pid_w * BLOCK_SIZE_W

    # Compute the output block indices
    oh = h // stride[0]
    ow = w // stride[1]

    # Load the input block
    input_block = tl.load(input_ptr + n * C * H * W + tl.arange(0, BLOCK_SIZE_C) * H * W + h * W + w, mask=(n < N) & (h < H) & (w < W))

    # Initialize the output block
    output_block = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_C, BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)

    # Compute the convolution
    for g in range(groups):
        for k_h in range(K_H):
            for k_w in range(K_W):
                input_h = h + k_h * dil_H - pad_H
                input_w = w + k_w * dil_W - pad_W
                if (input_h >= 0) and (input_h < H) and (input_w >= 0) and (input_w < W):
                    weight_block = tl.load(weight_ptr + g * O * Cg * K_H * K_W + tl.arange(0, O) * Cg * K_H * K_W + tl.arange(0, Cg) * K_H * K_W + k_h * K_W + k_w)
                    output_block += tl.dot(input_block, weight_block)

    # Add bias if provided
    if bias_ptr is not None:
        bias_block = tl.load(bias_ptr + tl.arange(0, O))
        output_block += bias_block

    # Add other if provided
    if other_ptr is not None:
        other_block = tl.load(other_ptr + tl.arange(0, Oth))
        output_block += alpha * other_block

    # Store the output block
    tl.store(out_ptr + n * O * OH * OW + oh * OW + ow, output_block, mask=(n < N) & (oh < OH) & (ow < OW))

import torch
import triton
import triton.language as tl

def conv2d_add(input, weight, bias=None, other=None, stride=1, padding=0, dilation=1, groups=1, alpha=1, out=None):
    # Ensure input and weight are contiguous
    input = input.contiguous()
    weight = weight.contiguous()

    # Extract shapes
    N, C, H, W = input.shape
    O, Cg, K_H, K_W = weight.shape

    # Validate shapes
    assert C % groups == 0, "in_channels must be divisible by groups"
    assert O % groups == 0, "out_channels must be divisible by groups"

    # Compute output dimensions
    if isinstance(padding, str):
        if padding == 'valid':
            padding = 0
        elif padding == 'same':
            padding = (K_H // 2, K_W // 2)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    OH = (H + 2 * padding[0] - dilation[0] * (K_H - 1) - 1) // stride[0] + 1
    OW = (W + 2 * padding[1] - dilation[1] * (K_W - 1) - 1) // stride[1] + 1

    # Initialize output tensor
    if out is None:
        out = torch.empty((N, O, OH, OW), device=input.device, dtype=input.dtype)

    # Launch the kernel
    grid = (triton.cdiv(N, 1), triton.cdiv(OH, 1), triton.cdiv(OW, 1))
    conv2d_add_kernel[grid](
        input, weight, bias, other, out,
        (N, C, H, W), (O, Cg, K_H, K_W), (O,) if bias is not None else (0,), (O,) if other is not None else (0,), (N, O, OH, OW),
        stride, padding, dilation, groups, alpha,
        BLOCK_SIZE_N=1, BLOCK_SIZE_C=1, BLOCK_SIZE_H=1, BLOCK_SIZE_W=1
    )

    return out

import torch
import torch.nn.functional as F

# Example inputs
input = torch.randn(2, 3, 10, 10, device='cuda')
weight = torch.randn(4, 3, 3, 3, device='cuda')
bias = torch.randn(4, device='cuda')
other = torch.randn(4, device='cuda')
stride = 1
padding = 1
dilation = 1
groups = 1
alpha = 1

# PyTorch reference
out_torch = F.conv2d(input, weight, bias, stride, padding, dilation, groups) + alpha * other.view(1, -1, 1, 1)

# Custom implementation
out_custom = conv2d_add(input, weight, bias, other, stride, padding, dilation, groups, alpha)

# Verify the results
print(torch.allclose(out_torch, out_custom))
