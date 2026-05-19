import triton
import triton.language as tl

@triton.jit
def combined_activation_kernel(
    input_ptr, weight1_ptr, weight2_ptr, bias_ptr, output_ptr,
    input_stride_0, input_stride_1, input_stride_2,
    weight1_stride_0, weight1_stride_1,
    weight2_stride_0, weight2_stride_1,
    bias_stride_0, bias_stride_1,
    output_stride_0, output_stride_1, output_stride_2,
    batch_size, N, D_in, D_out,
    BLOCK_SIZE: tl.constexpr
):
    # Compute the position of the current block
    pid = tl.program_id(axis=0)
    batch_idx = pid // (N * D_out)
    n_idx = (pid % (N * D_out)) // D_out
    d_out_idx = (pid % (N * D_out)) % D_out

    # Compute the starting index of the block
    input_offset = batch_idx * input_stride_0 + n_idx * input_stride_1
    weight1_offset = d_out_idx * weight1_stride_0
    weight2_offset = d_out_idx * weight2_stride_0
    bias_offset = d_out_idx * bias_stride_0
    output_offset = batch_idx * output_stride_0 + n_idx * output_stride_1 + d_out_idx * output_stride_2

    # Load the input and weight1
    input_block = tl.load(input_ptr + input_offset + tl.arange(0, BLOCK_SIZE) * input_stride_2)
    weight1_block = tl.load(weight1_ptr + weight1_offset + tl.arange(0, BLOCK_SIZE) * weight1_stride_1)

    # Perform matrix multiplication
    matmul_result = tl.dot(input_block, weight1_block)

    # Apply sigmoid
    sigmoid_result = 1 / (1 + tl.exp(-matmul_result))

    # Apply tanh
    tanh_result = (tl.exp(sigmoid_result) - tl.exp(-sigmoid_result)) / (tl.exp(sigmoid_result) + tl.exp(-sigmoid_result))

    # Load weight2 and bias
    weight2_block = tl.load(weight2_ptr + weight2_offset)
    bias_block = tl.load(bias_ptr + bias_offset)

    # Perform element-wise multiplication and addition
    final_result = tanh_result * weight2_block + bias_block

    # Store the result
    tl.store(output_ptr + output_offset, final_result)

import torch
import triton
import triton.language as tl

def combined_activation(input, weight1, weight2, bias, *, out=None):
    # Ensure input and weight1 are compatible for matrix multiplication
    batch_size, N, D_in = input.shape
    D_out = weight1.shape[1]
    
    # Ensure weight2 and bias are broadcastable to the output shape
    weight2 = weight2.expand(batch_size, N, D_out)
    bias = bias.expand(batch_size, N, D_out)
    
    # Allocate output tensor if not provided
    if out is None:
        out = torch.empty((batch_size, N, D_out), device=input.device, dtype=input.dtype)
    
    # Launch the Triton kernel
    grid = (batch_size * N * D_out, )
    combined_activation_kernel[grid](
        input, weight1, weight2, bias, out,
        input.stride(0), input.stride(1), input.stride(2),
        weight1.stride(0), weight1.stride(1),
        weight2.stride(0), weight2.stride(1),
        bias.stride(0), bias.stride(1),
        out.stride(0), out.stride(1), out.stride(2),
        batch_size, N, D_in, D_out,
        BLOCK_SIZE=D_in
    )
    
    return out

# Example data
batch_size, N, D_in, D_out = 2, 3, 4, 5
input = torch.randn(batch_size, N, D_in, device='cuda')
weight1 = torch.randn(D_in, D_out, device='cuda')
weight2 = torch.randn(1, 1, D_out, device='cuda')
bias = torch.randn(1, 1, D_out, device='cuda')

# Call the function
output = combined_activation(input, weight1, weight2, bias)

# Print the output
print(output)
