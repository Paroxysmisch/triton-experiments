import torch
import triton
import triton.language as tl

def grid_sample(input, grid, mode='bilinear', padding_mode='zeros', align_corners=False):
    # Determine input and grid shapes
    n, c, h_in, w_in = input.shape
    _, h_out, w_out, _ = grid.shape

    # Prepare device
    device = input.device

    # Allocate output tensor
    output = torch.zeros((n, c, h_out, w_out), device=device, dtype=input.dtype)

    # Define Triton kernel arguments
    grid_sample_kernel_config = triton.autotune([triton.next_power_of_two(h_out), triton.next_power_of_two(w_out)], [32])
    grid_sample_kernel[
        grid_sample_kernel_config,
        (n, c, h_out, w_out),
        (n, c, h_out, w_out)
    ](
        input.contiguous().data_ptr(),
        grid.contiguous().data_ptr(),
        output.data_ptr(),
        input.shape,
        grid.shape,
        mode,
        padding_mode,
        align_corners,
        n,
        c,
        h_in,
        w_in,
        h_out=h_out,
        w_out=w_out,
        dtype=input.dtype,
    )

    return output
