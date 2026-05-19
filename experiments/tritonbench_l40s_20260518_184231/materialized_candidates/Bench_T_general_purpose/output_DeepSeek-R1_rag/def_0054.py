import torch
import triton
import triton.language as tl

def _pair(x):
    if isinstance(x, int):
        return (x, x)
    elif isinstance(x, tuple):
        assert len(x) == 2, "Pair should be a tuple of two elements"
        return x
    else:
        raise ValueError("Padding should be either int or tuple")

@triton.jit
def gelu_conv2d_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    dilation_h,
    dilation_w,
    groups,
    in_channels,
    out_channels,
    kH,
    kW,
    iH,
    iW,
    oH,
    oW,
    approximate,
    BLOCK_SIZE: tl.constexpr,
):
    # Extract program IDs
    n = tl.program_id(0)
    oc = tl.program_id(1)
    pid = tl.program_id(2)

    # Compute spatial offsets for this block
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    ow = offsets % oW
    oh = offsets // oW
    mask = (oh < oH) & (ow < oW)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    # Determine group info
    out_channels_per_group = out_channels // groups
    group_id = oc // out_channels_per_group
    in_channels_per_group = in_channels // groups
    ic_start = group_id * in_channels_per_group

    # Loop over input channels in the group
    for icg in range(in_channels_per_group):
        ic = ic_start + icg
        # Loop over kernel elements
        for kh in range(kH):
            for kw in range(kW):
                # Calculate input positions with dilation and padding
                ih = oh * stride_h + kh * dilation_h - pad_h
                iw = ow * stride_w + kw * dilation_w - pad_w

                # Check if input indices are within bounds
                input_mask = (ih >= 0) & (ih < iH) & (iw >= 0) & (iw < iW) & mask
                input_idx = n * in_channels * iH * iW + ic * iH * iW + ih * iW + iw
                input_val = tl.load(input_ptr + input_idx, mask=input_mask, other=0.0)

                # Load weight value
                weight_idx = oc * (in_channels_per_group * kH * kW) + icg * (kH * kW) + kh * kW + kw
                weight_val = tl.load(weight_ptr + weight_idx)

                # Accumulate
                acc += input_val * weight_val

    # Add bias if present
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + oc)
        acc += bias_val

    # Apply GELU activation
    if approximate == 'none':
        # GELU with erf
        cdf = 0.5 * (1.0 + tl.math.erf(acc / tl.math.sqrt(2.0)))
        output_val = acc * cdf
    elif approximate == 'tanh':
        # Approximate GELU with tanh
        sqrt_2_over_pi = tl.math.sqrt(2.0 / tl.math.pi)
        x = acc
        x_cubed = x * x * x
        inner = sqrt_2_over_pi * (x + 0.044715 * x_cubed)
        tanh_inner = tl.math.tanh(inner)
        output_val = 0.5 * x * (1.0 + tanh_inner)
    else:
        # Invalid approximate method, default to identity
        output_val = acc

    # Compute output indices and store
    output_idx = n * out_channels * oH * oW + oc * oH * oW + oh * oW + ow
    tl.store(output_ptr + output_idx, output_val, mask=mask)

def gelu_conv2d(input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None,
                stride: Union[int, Tuple[int, int]] = 1, padding: Union[int, Tuple[int, int], str] = 0,
                dilation: Union[int, Tuple[int, int]] = 1, groups: int = 1,
                approximate: str = 'none', out: Optional[torch.Tensor] = None) -> torch.Tensor:
    # Check input and weight dimensions
    assert input.dim() == 4, "Input must be 4D (N, C, H, W)"
    assert weight.dim() == 4, "Weight must be 4D (O, C//groups, kH, kW)"
    if bias is not None:
        assert bias.dim() == 1, "Bias must be 1D (O)"
        assert bias.size(0) == weight.size(0), "Bias size must match out_channels"

    # Ensure tensors are contiguous and on CUDA
    input = input.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    N, in_channels, iH, iW = input.shape
    out_channels, in_channels_per_group, kH, kW = weight.shape
    assert in_channels_per_group * groups == in_channels, "in_channels must be divisible by groups"
    assert out_channels % groups == 0, "out_channels must be divisible by groups"

    # Process stride, padding, dilation
    stride_h, stride_w = _pair(stride)
    dilation_h, dilation_w = _pair(dilation)

    # Handle padding
    if isinstance(padding, str):
        if padding.lower() == 'same':
            pad_h = (kH - 1) * dilation_h // 2
            pad_w = (kW - 1) * dilation_w // 2
        elif padding.lower() == 'valid':
            pad_h, pad_w = 0, 0
        else:
            raise ValueError(f"Unsupported padding mode: {padding}")
    else:
        pad_h, pad_w = _pair(padding)

    # Calculate output dimensions
    oH = (iH + 2 * pad_h - dilation_h * (kH - 1) - 1) // stride_h + 1
    oW = (iW + 2 * pad_w - dilation_w * (kW - 1) - 1) // stride_w + 1
    assert oH > 0 and oW > 0, f"Output dimensions must be positive, got oH={oH}, oW={oW}"

    # Allocate output tensor if not provided
    if out is None:
        out = torch.empty((N, out_channels, oH, oW), device=input.device, dtype=input.dtype)
    else:
        assert out.shape == (N, out_channels, oH, oW), "Output tensor has incorrect shape"

    # Ensure CUDA tensors
    assert input.is_cuda and weight.is_cuda, "Input and weight must be on CUDA"
    if bias is not None:
        assert bias.is_cuda, "Bias must be on CUDA"

    # Kernel parameters
    BLOCK_SIZE = 256
    num_blocks = (oH * oW + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (N, out_channels, num_blocks)

    # Launch kernel
    gelu_conv2d_kernel[grid](
        input_ptr=input,
        weight_ptr=weight,
        bias_ptr=bias,
        output_ptr=out,
        stride_h=stride_h,
        stride_w=stride_w,
        pad_h=pad_h,
        pad_w=pad_w,
        dilation_h=dilation_h,
        dilation_w=dilation_w,
        groups=groups,
        in_channels=in_channels,
        out_channels=out_channels,
        kH=kH,
        kW=kW,
        iH=iH,
        iW=iW,
        oH=oH,
        oW=oW,
        approximate=approximate,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out
