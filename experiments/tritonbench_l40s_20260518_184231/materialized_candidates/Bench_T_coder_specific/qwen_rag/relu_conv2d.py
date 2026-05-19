import triton
import triton.language as tl

@triton.jit
def conv2d_kernel(
    input_ptr, weight_ptr, output_ptr, bias_ptr, input_shape, weight_shape, stride, padding, dilation, groups, BLOCK_SIZE: tl.constexpr
):
    # Unpack shapes
    minibatch, in_channels, iH, iW = input_shape
    out_channels, _, kH, kW = weight_shape

    # Calculate effective spatial dimensions after padding
    effective_iH = iH + 2 * padding
    effective_iW = iW + 2 * padding

    # Calculate output dimensions
    oH = (effective_iH - dilation * (kH - 1)) // stride
    oW = (effective_iW - dilation * (kW - 1)) // stride

    # Get program ID
    n = tl.program_id(0)  # Output channel
    c = tl.program_id(1)  # Input channel
    h = tl.program_id(2)  # Output height
    w = tl.program_id(3)  # Output width

    # Calculate input indices
    ih_base = h * stride - padding
    iw_base = w * stride - padding

    # Initialize result
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    # Loop over kernel elements
    for kh in range(kH):
        for kw in range(kW):
            # Calculate effective kernel index considering dilation
            kh_eff = kh * dilation
            kw_eff = kw * dilation

            # Calculate input indices
            ih = ih_base + kh_eff
            iw = iw_base + kw_eff

            # Load input and weight
            input_val = tl.load(input_ptr + ((n * in_channels + c) * effective_iH + ih) * effective_iW + iw, mask=(ih < effective_iH) & (iw < effective_iW), other=0.0)
            weight_val = tl.load(weight_ptr + ((n * groups + c // groups) * kH + kh_eff) * kW + kw_eff)

            # Accumulate result
            acc += input_val * weight_val

    # Add bias if provided
    if bias_ptr is not None:
        acc += tl.load(bias_ptr + n)

    # Store result
    tl.store(output_ptr + (n * minibatch * oH + h) * oW + w, acc)
