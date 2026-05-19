import triton
import triton.language as tl

@triton.jit
def adaptive_avg_pool2d_kernel(
    input_ptr,  # *input tensor
    output_ptr,  # *output tensor
    N,  # batch size
    C,  # number of channels
    H_in,  # input height
    W_in,  # input width
    H_out,  # output height
    W_out,  # output width
    BLOCK_SIZE_C: tl.constexpr,  # block size for channels
    BLOCK_SIZE_H: tl.constexpr,  # block size for height
    BLOCK_SIZE_W: tl.constexpr,  # block size for width
):
    # Compute the output position for this thread
    pid = tl.program_id(axis=0)
    n = pid // (C * H_out * W_out)
    c = (pid % (C * H_out * W_out)) // (H_out * W_out)
    h_out = (pid % (H_out * W_out)) // W_out
    w_out = pid % W_out

    # Compute the input region for this output position
    h_start = h_out * H_in // H_out
    h_end = (h_out + 1) * H_in // H_out
    w_start = w_out * W_in // W_out
    w_end = (w_out + 1) * W_in // W_out

    # Compute the average value for this output position
    sum_val = 0.0
    count = 0
    for h in range(h_start, h_end):
        for w in range(w_start, w_end):
            input_offset = n * C * H_in * W_in + c * H_in * W_in + h * W_in + w
            sum_val += tl.load(input_ptr + input_offset)
            count += 1

    avg_val = sum_val / count

    # Write the result to the output tensor
    output_offset = n * C * H_out * W_out + c * H_out * W_out + h_out * W_out + w_out
    tl.store(output_ptr + output_offset, avg_val)

import torch
import triton
import triton.language as tl

def adaptive_avg_pool2d(input: torch.Tensor, output_size) -> torch.Tensor:
    # Determine the output size
    if isinstance(output_size, int):
        H_out = W_out = output_size
    elif isinstance(output_size, tuple) and len(output_size) == 2:
        H_out, W_out = output_size
    else:
        raise ValueError("output_size must be an integer or a tuple of two integers")

    # Determine the input shape
    N, C, H_in, W_in = input.shape if input.dim() == 4 else (1, *input.shape)

    # Determine the output shape
    if H_out is None:
        H_out = H_in
    if W_out is None:
        W_out = W_in

    # Create the output tensor
    output = torch.empty((N, C, H_out, W_out), device=input.device, dtype=input.dtype)

    # Define the grid and block sizes
    grid = (N * C * H_out * W_out, )
    block = (1, )

    # Launch the kernel
    adaptive_avg_pool2d_kernel[grid, block](
        input,  # *input tensor
        output,  # *output tensor
        N,  # batch size
        C,  # number of channels
        H_in,  # input height
        W_in,  # input width
        H_out,  # output height
        W_out,  # output width
        BLOCK_SIZE_C=1,  # block size for channels
        BLOCK_SIZE_H=1,  # block size for height
        BLOCK_SIZE_W=1,  # block size for width
    )

    # Return the output tensor
    return output
