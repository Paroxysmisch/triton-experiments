import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 128, 'RBLOCK': 16}),
        triton.Config({'XBLOCK': 256, 'RBLOCK': 32}),
    ],
    key=['xnumel', 'rnumel']
)
@triton.jit
def triton_red_fused_native_layer_norm_no_welford(
    in_ptr0, in_out_ptr0, in_out_ptr1, in_ptr1, in_ptr2, out_ptr0,
    xnumel, rnumel, stride_x, stride_r, scale, shift,
    BLOCK_SIZE: tl.constexpr
):
    # Define the program id
    pid = tl.program_id(axis=0)

    # Define the range of elements this program will handle
    x_start = pid * BLOCK_SIZE
    x_end = min(x_start + BLOCK_SIZE, xnumel)

    # Initialize mean and variance accumulators
    mean_acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    var_acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # Compute mean
    for r in range(0, rnumel):
        # Load input data
        idx = x_start + tl.arange(0, BLOCK_SIZE)
        input_data = tl.load(in_ptr0 + idx * stride_x + r * stride_r, mask=idx < x_end, other=0.0)
        
        # Accumulate sum
        mean_acc += input_data

    # Compute mean
    mean = mean_acc / rnumel

    # Store mean in the output buffer
    tl.store(in_out_ptr0 + x_start, mean, mask=idx < x_end)

    # Compute variance
    for r in range(0, rnumel):
        # Load input data
        idx = x_start + tl.arange(0, BLOCK_SIZE)
        input_data = tl.load(in_ptr0 + idx * stride_x + r * stride_r, mask=idx < x_end, other=0.0)
        
        # Accumulate variance
        diff = input_data - mean
        var_acc += diff * diff

    # Compute variance and inverse standard deviation
    variance = var_acc / rnumel
    inv_stddev = tl.libdevice.rsqrt(variance + 1e-5)

    # Store inverse standard deviation in the output buffer
    tl.store(in_out_ptr1 + x_start, inv_stddev, mask=idx < x_end)

    # Normalize the input data and apply scale and shift
    for r in range(0, rnumel):
        # Load input data
        idx = x_start + tl.arange(0, BLOCK_SIZE)
        input_data = tl.load(in_ptr0 + idx * stride_x + r * stride_r, mask=idx < x_end, other=0.0)
        
        # Normalize
        normalized = (input_data - mean) * inv_stddev

        # Apply scale and shift
        scaled_shifted = normalized * scale + shift

        # Store the result
        tl.store(out_ptr0 + idx * stride_x + r * stride_r, scaled_shifted, mask=idx < x_end)

def fused_native_layer_norm_no_welford(
    input_data, scale, shift, xnumel, rnumel, stride_x, stride_r, device
):
    # Allocate output buffers
    mean_output = torch.empty(xnumel, device=device, dtype=torch.float32)
    inv_stddev_output = torch.empty(xnumel, device=device, dtype=torch.float32)
    normalized_output = torch.empty_like(input_data)

    # Define grid
    grid = lambda META: (triton.cdiv(xnumel, META['XBLOCK']),)

    # Launch kernel
    triton_red_fused_native_layer_norm_no_welford[grid](
        input_data, mean_output, inv_stddev_output, scale, shift, normalized_output,
        xnumel, rnumel, stride_x, stride_r,
        BLOCK_SIZE=128,  # This should match one of the autotune configurations
        num_warps=4,
    )

    return normalized_output, mean_output, inv_stddev_output
