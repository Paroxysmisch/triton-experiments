import triton
import triton.language as tl

@triton.jit
def rmsnorm_triton(
    x_ptr,  # Pointer to the input tensor
    rms_w_ptr,  # Pointer to the RMS normalization weights
    output_ptr,  # Pointer to the output tensor
    stride_x_batch,  # Stride for the batch dimension of x
    stride_x_m,  # Stride for the second dimension (M) of x
    stride_x_n,  # Stride for the last dimension (N) of x
    stride_rms_w_n,  # Stride for the last dimension (N) of rms_w
    stride_output_batch,  # Stride for the batch dimension of output
    stride_output_m,  # Stride for the second dimension (M) of output
    stride_output_n,  # Stride for the last dimension (N) of output
    N_SIZE,  # Size of the last dimension (N)
    eps,  # Epsilon for numerical stability
    BLOCK_N_SIZE: tl.constexpr  # Block size for the last dimension
):
    pid_batch = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)

    # Pointers to the start of the batch and row in the input tensor
    x_batch_ptr = x_ptr + pid_batch * stride_x_batch + pid_m * stride_x_m
    output_batch_ptr = output_ptr + pid_batch * stride_output_batch + pid_m * stride_output_m

    # Initialize the sum of squares
    sum_of_squares = tl.zeros((1,), dtype=tl.float32)

    # Iterate over chunks of size BLOCK_N_SIZE
    for n in range(0, N_SIZE, BLOCK_N_SIZE):
        # Load a chunk of the input tensor
        x_chunk = tl.load(x_batch_ptr + n * stride_x_n, mask=n + tl.arange(0, BLOCK_N_SIZE) < N_SIZE, other=0.0)
        
        # Compute the sum of squares for the chunk
        sum_of_squares += tl.sum(x_chunk * x_chunk, axis=0)

    # Compute the variance and reciprocal of the standard deviation
    variance = sum_of_squares / N_SIZE
    rstd = 1.0 / tl.sqrt(variance + eps)

    # Normalize the input and scale by the weights
    for n in range(0, N_SIZE, BLOCK_N_SIZE):
        # Load a chunk of the input tensor
        x_chunk = tl.load(x_batch_ptr + n * stride_x_n, mask=n + tl.arange(0, BLOCK_N_SIZE) < N_SIZE, other=0.0)
        
        # Load the corresponding weights
        rms_w_chunk = tl.load(rms_w_ptr + n * stride_rms_w_n, mask=n + tl.arange(0, BLOCK_N_SIZE) < N_SIZE, other=0.0)
        
        # Normalize and scale
        output_chunk = (x_chunk * rstd) * rms_w_chunk
        
        # Store the result back to the output tensor
        tl.store(output_batch_ptr + n * stride_output_n, output_chunk, mask=n + tl.arange(0, BLOCK_N_SIZE) < N_SIZE)

import torch

def rmsnorm_triton_wrapper(x, rms_w, eps=1e-6, BLOCK_N_SIZE=128):
    # Get the dimensions of the input tensor
    B, M, N = x.shape

    # Ensure the input tensor is on the GPU
    x = x.cuda()
    rms_w = rms_w.cuda()

    # Initialize the output tensor
    output = torch.empty_like(x)

    # Define the grid and block dimensions
    grid = (B, M)

    # Launch the kernel
    rmsnorm_triton[grid](
        x,  # Pointer to the input tensor
        rms_w,  # Pointer to the RMS normalization weights
        output,  # Pointer to the output tensor
        x.stride(0),  # Stride for the batch dimension of x
        x.stride(1),  # Stride for the second dimension (M) of x
        x.stride(2),  # Stride for the last dimension (N) of x
        rms_w.stride(0),  # Stride for the last dimension (N) of rms_w
        output.stride(0),  # Stride for the batch dimension of output
        output.stride(1),  # Stride for the second dimension (M) of output
        output.stride(2),  # Stride for the last dimension (N) of output
        N,  # Size of the last dimension (N)
        eps,  # Epsilon for numerical stability
        BLOCK_N_SIZE  # Block size for the last dimension
    )

    return output
