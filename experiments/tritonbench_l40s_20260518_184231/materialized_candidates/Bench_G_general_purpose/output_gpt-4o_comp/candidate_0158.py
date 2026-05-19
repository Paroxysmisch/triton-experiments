import triton
import triton.language as tl

@triton.jit
def rmsnorm_triton(x_ptr, rms_w_ptr, output_ptr, stride_xm, stride_xn, stride_w, stride_om, stride_on,
                   N_SIZE: tl.constexpr, eps: tl.constexpr, BLOCK_N_SIZE: tl.constexpr):
    pid_batch = tl.program_id(0)
    pid_m = tl.program_id(1)

    # Offsets for the current batch and row
    x_offset = pid_batch * stride_xm + pid_m * stride_xn
    w_offset = pid_m * stride_w
    o_offset = pid_batch * stride_om + pid_m * stride_on

    # Initialize sum of squares
    sum_squares = tl.zeros((BLOCK_N_SIZE,), dtype=tl.float32)

    # Iterate over chunks of size BLOCK_N_SIZE
    for start_n in range(0, N_SIZE, BLOCK_N_SIZE):
        # Load a block of data
        x_block = tl.load(x_ptr + x_offset + start_n, mask=start_n + tl.arange(0, BLOCK_N_SIZE) < N_SIZE)
        # Accumulate sum of squares
        sum_squares += x_block * x_block

    # Calculate variance and reciprocal standard deviation
    variance = tl.sum(sum_squares) / N_SIZE
    rstd = 1.0 / tl.sqrt(variance + eps)

    # Normalize, scale by weights, and store the result
    for start_n in range(0, N_SIZE, BLOCK_N_SIZE):
        # Load a block of data
        x_block = tl.load(x_ptr + x_offset + start_n, mask=start_n + tl.arange(0, BLOCK_N_SIZE) < N_SIZE)
        # Load weights
        w_block = tl.load(rms_w_ptr + w_offset + start_n, mask=start_n + tl.arange(0, BLOCK_N_SIZE) < N_SIZE)
        # Normalize and scale
        norm_block = (x_block * rstd) * w_block
        # Store the result
        tl.store(output_ptr + o_offset + start_n, norm_block, mask=start_n + tl.arange(0, BLOCK_N_SIZE) < N_SIZE)

import torch

def rmsnorm_triton_wrapper(x, rms_w, eps, BLOCK_N_SIZE):
    # Extract dimensions
    batch_size, M, N_SIZE = x.shape

    # Initialize output tensor
    output = torch.empty_like(x)

    # Define grid size
    grid = (batch_size, M)

    # Launch the Triton kernel
    rmsnorm_triton[grid](
        x_ptr=x,
        rms_w_ptr=rms_w,
        output_ptr=output,
        stride_xm=x.stride(0),
        stride_xn=x.stride(1),
        stride_w=rms_w.stride(0),
        stride_om=output.stride(0),
        stride_on=output.stride(1),
        N_SIZE=N_SIZE,
        eps=eps,
        BLOCK_N_SIZE=BLOCK_N_SIZE
    )

    return output
