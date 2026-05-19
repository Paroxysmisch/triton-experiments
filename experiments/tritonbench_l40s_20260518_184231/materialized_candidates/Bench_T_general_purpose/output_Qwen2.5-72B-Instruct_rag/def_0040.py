import triton
import triton.language as tl

@triton.jit
def batch_norm_kernel(
    input_ptr, running_mean_ptr, running_var_ptr, weight_ptr, bias_ptr,
    output_ptr, N, C, L, training, momentum, eps,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_C: tl.constexpr, BLOCK_SIZE_L: tl.constexpr
):
    pid = tl.program_id(axis=0)
    batch_size = N
    num_channels = C
    seq_length = L

    # Compute the block indices
    block_n = pid // (num_channels * seq_length)
    block_c = (pid % (num_channels * seq_length)) // seq_length
    block_l = (pid % (num_channels * seq_length)) % seq_length

    # Compute the offsets for the input and output
    input_offset = block_n * num_channels * seq_length + block_c * seq_length + block_l
    output_offset = block_n * num_channels * seq_length + block_c * seq_length + block_l

    # Load the input and running statistics
    input = tl.load(input_ptr + input_offset)
    running_mean = tl.load(running_mean_ptr + block_c)
    running_var = tl.load(running_var_ptr + block_c)

    # Load the weight and bias if they are provided
    weight = tl.load(weight_ptr + block_c) if weight_ptr is not None else 1.0
    bias = tl.load(bias_ptr + block_c) if bias_ptr is not None else 0.0

    # Compute the normalized value
    normalized = (input - running_mean) * tl.rsqrt(running_var + eps)
    normalized = normalized * weight + bias

    # Apply the sigmoid activation function
    output = 1.0 / (1.0 + tl.exp(-normalized))

    # Store the output
    tl.store(output_ptr + output_offset, output)

    # Update running statistics if in training mode
    if training:
        # Compute the mean and variance of the current batch
        batch_mean = tl.sum(input, axis=1) / seq_length
        batch_var = tl.sum((input - batch_mean) ** 2, axis=1) / seq_length

        # Update the running mean and variance
        running_mean = (1 - momentum) * running_mean + momentum * batch_mean
        running_var = (1 - momentum) * running_var + momentum * batch_var

        # Store the updated running statistics
        tl.store(running_mean_ptr + block_c, running_mean)
        tl.store(running_var_ptr + block_c, running_var)

import torch
import triton

def sigmoid_batch_norm(input, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-5):
    # Ensure the input is a contiguous tensor
    input = input.contiguous()
    N, C = input.shape[:2]
    L = input.shape[2] if len(input.shape) == 3 else 1

    # Ensure the running_mean and running_var are contiguous
    running_mean = running_mean.contiguous()
    running_var = running_var.contiguous()

    # Ensure the weight and bias are contiguous if provided
    if weight is not None:
        weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    # Allocate output tensor
    output = torch.empty_like(input)

    # Define the grid and block sizes
    grid = (N * C * L, 1, 1)
    block = (1, 1, 1)

    # Launch the Triton kernel
    batch_norm_kernel[grid, block](
        input, running_mean, running_var, weight, bias,
        output, N, C, L, training, momentum, eps,
        BLOCK_SIZE_N=1, BLOCK_SIZE_C=1, BLOCK_SIZE_L=1
    )

    return output
