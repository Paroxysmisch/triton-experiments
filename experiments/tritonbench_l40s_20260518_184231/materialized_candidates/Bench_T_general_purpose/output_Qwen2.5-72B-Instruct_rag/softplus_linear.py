import torch
import triton
import triton.language as tl

# Triton kernel for the softplus_linear operation
@triton.jit
def softplus_linear_kernel(
    output_ptr, input_ptr, weight_ptr, bias_ptr, input_stride, weight_stride, output_stride,
    n_inputs, n_outputs, beta, threshold, BLOCK_SIZE: tl.constexpr
):
    # Compute the linear transformation
    pid = tl.program_id(0)
    batch_start = pid * BLOCK_SIZE
    batch_end = batch_start + BLOCK_SIZE

    # Load the input and weight matrices
    input_offsets = tl.arange(0, BLOCK_SIZE)
    weight_offsets = tl.arange(0, n_outputs)
    input_ptrs = input_ptr + batch_start * input_stride + input_offsets
    weight_ptrs = weight_ptr + weight_offsets * weight_stride

    # Load input and weight data
    input_data = tl.load(input_ptrs, mask=input_offsets < n_inputs, other=0.0)
    weight_data = tl.load(weight_ptrs, mask=weight_offsets < n_outputs, other=0.0)

    # Compute the linear transformation
    linear_output = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for i in range(n_inputs):
        linear_output += input_data[i] * weight_data

    # Add bias if provided
    if bias_ptr is not None:
        bias_data = tl.load(bias_ptr, mask=weight_offsets < n_outputs, other=0.0)
        linear_output += bias_data

    # Apply the Softplus activation
    softplus_output = tl.where(linear_output * beta > threshold, linear_output, (1 / beta) * tl.log(1 + tl.exp(beta * linear_output)))

    # Write the result back to global memory
    output_ptrs = output_ptr + batch_start * output_stride + weight_offsets
    tl.store(output_ptrs, softplus_output, mask=weight_offsets < n_outputs)

# Wrapper function for the softplus_linear operation
def softplus_linear(input, weight, bias=None, beta=1, threshold=20):
    # Validate input dimensions
    assert input.dim() == 2, "Input tensor must be 2D"
    assert weight.dim() == 2, "Weight tensor must be 2D"
    assert input.size(1) == weight.size(0), "Input and weight dimensions must match"

    # Get input and output dimensions
    n_inputs, n_outputs = input.size(1), weight.size(1)
    n_batches = input.size(0)

    # Allocate output tensor
    output = torch.empty((n_batches, n_outputs), device=input.device, dtype=input.dtype)

    # Determine block size
    BLOCK_SIZE = triton.next_power_of_2(n_inputs)

    # Determine number of warps
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16

    # Enqueue the kernel
    softplus_linear_kernel[(n_batches,)](
        output, input, weight, bias,
        input.stride(0), weight.stride(0), output.stride(0),
        n_inputs, n_outputs, beta, threshold,
        num_warps=num_warps, BLOCK_SIZE=BLOCK_SIZE
    )

    return output
