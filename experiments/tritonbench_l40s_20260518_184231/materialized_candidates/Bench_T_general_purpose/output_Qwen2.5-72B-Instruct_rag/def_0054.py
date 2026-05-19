import torch
import triton
import triton.language as tl

# Triton kernel for 2D convolution followed by GELU activation
@triton.jit
def gelu_conv2d_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr, 
    input_shape, weight_shape, output_shape, 
    stride, padding, dilation, groups, 
    approximate, 
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_C: tl.constexpr, BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_W: tl.constexpr
):
    # Extract shapes
    N, C, H, W = input_shape
    K, Cg, R, S = weight_shape
    N, K, Ho, Wo = output_shape

    # Extract convolution parameters
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    dilation_h, dilation_w = dilation

    # Compute the grid and block indices
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_ho = tl.program_id(2)
    pid_wo = tl.program_id(3)

    # Compute the block start indices
    ho_start = pid_ho * BLOCK_SIZE_H
    wo_start = pid_wo * BLOCK_SIZE_W

    # Compute the block end indices
    ho_end = min(ho_start + BLOCK_SIZE_H, Ho)
    wo_end = min(wo_start + BLOCK_SIZE_W, Wo)

    # Initialize the output block
    output_block = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_C, BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)

    # Iterate over the input channels and kernel positions
    for r in range(R):
        for s in range(S):
            for c in range(Cg):
                # Compute the input indices
                h_start = ho_start * stride_h - pad_h + r * dilation_h
                w_start = wo_start * stride_w - pad_w + s * dilation_w

                # Compute the input block
                input_block = tl.load(
                    input_ptr + (pid_n * C * H * W + c * H * W + h_start * W + w_start) * BLOCK_SIZE_H * BLOCK_SIZE_W,
                    mask=(h_start + tl.arange(0, BLOCK_SIZE_H) < H) & (w_start + tl.arange(0, BLOCK_SIZE_W) < W),
                    other=0.0
                )

                # Compute the weight block
                weight_block = tl.load(
                    weight_ptr + (pid_k * Cg * R * S + c * R * S + r * S + s) * BLOCK_SIZE_H * BLOCK_SIZE_W
                )

                # Perform the convolution
                output_block += input_block * weight_block

    # Apply bias if provided
    if bias_ptr is not None:
        bias_block = tl.load(bias_ptr + pid_k * BLOCK_SIZE_C)
        output_block += bias_block

    # Apply GELU activation
    if approximate == 'none':
        output_block = output_block * tl.erf(output_block / tl.sqrt(tl.float32(2.0)))
    elif approximate == 'tanh':
        output_block = 0.5 * output_block * (1 + tl.tanh(tl.sqrt(tl.float32(2.0 / tl.pi)) * (output_block + 0.044715 * output_block * output_block * output_block)))

    # Store the output block
    tl.store(
        output_ptr + (pid_n * K * Ho * Wo + pid_k * Ho * Wo + ho_start * Wo + wo_start) * BLOCK_SIZE_H * BLOCK_SIZE_W,
        output_block,
        mask=(ho_start + tl.arange(0, BLOCK_SIZE_H) < Ho) & (wo_start + tl.arange(0, BLOCK_SIZE_W) < Wo)
    )

# Python wrapper function
def gelu_conv2d(input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None, 
                stride: Union[int, Tuple[int, int]] = 1, padding: Union[int, Tuple[int, int], str] = 0, 
                dilation: Union[int, Tuple[int, int]] = 1, groups: int = 1, 
                approximate: str = 'none', out: Optional[torch.Tensor] = None) -> torch.Tensor:
    # Ensure input and weight are on the same device
    device = input.device
    assert device.type == "cuda", "Input tensor must be on a CUDA device for triton ops."
    assert input.device == weight.device, "Input and weight must be on the same CUDA device."

    # Ensure bias is on the same device if provided
    if bias is not None:
        assert bias.device == device, "Bias must be on the same CUDA device as input and weight."

    # Convert stride, padding, and dilation to tuples if they are single numbers
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    # Compute output shape
    N, C, H, W = input.shape
    K, Cg, R, S = weight.shape
    Ho = (H + 2 * padding[0] - dilation[0] * (R - 1) - 1) // stride[0] + 1
    Wo = (W + 2 * padding[1] - dilation[1] * (S - 1) - 1) // stride[1] + 1

    # Allocate output tensor if not provided
    if out is None:
        out = torch.empty((N, K, Ho, Wo), device=device, dtype=input.dtype)

    # Define block and grid sizes
    BLOCK_SIZE_N = 1
    BLOCK_SIZE_C = 1
    BLOCK_SIZE_H = 16
    BLOCK_SIZE_W = 16

    grid = (N, K, (Ho + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H, (Wo + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W)

    # Launch the kernel
    gelu_conv2d_kernel[grid](
        input, weight, bias, out,
        (N, C, H, W), (K, Cg, R, S), (N, K, Ho, Wo),
        stride, padding, dilation, groups,
        approximate,
        BLOCK_SIZE_N, BLOCK_SIZE_C, BLOCK_SIZE_H, BLOCK_SIZE_W
    )

    return out
