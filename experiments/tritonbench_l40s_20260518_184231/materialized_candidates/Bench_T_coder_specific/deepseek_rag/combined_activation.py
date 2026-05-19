import torch
import triton
import triton.language as tl

@triton.jit
def combined_activation_kernel(
    input_ptr,
    weight1_ptr,
    weight2_ptr,
    bias_ptr,
    output_ptr,
    N,
    D_in,
    D_out,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the linear index of the current thread
    idx = tl.program_id(0)

    # Load the input and weight matrices
    input_matrix = tl.load(input_ptr + idx * D_in, (BLOCK_SIZE, D_in))
    weight1_matrix = tl.load(weight1_ptr, (D_in, D_out))
    weight2_matrix = tl.load(weight2_ptr, (D_out,))
    bias_vector = tl.load(bias_ptr, (D_out,))

    # Perform the matrix multiplication and activation functions
    intermediate = tl.dot(input_matrix, weight1_matrix)
    activation = tl.tanh(intermediate)
    output_vector = tl.sigmoid(activation)
    elementwise_mul = tl.multiply(output_vector, weight2_matrix)

    # Add the bias and store the result
    output_vector = tl.add(elementwise_mul, bias_vector)
    tl.store(output_ptr + idx * D_out, output_vector)

def combined_activation(input, weight1, weight2, bias, out=None):
    # Check the input and weight tensors' dimensions
    assert input.shape[-1] == weight1.shape[0]
    assert weight2.shape[0] == bias.shape[0]

    # Create the output tensor if it's not provided
    if out is None:
        out = torch.empty_like(input)

    # Compute the grid size
    grid = lambda META: (triton.cdiv(input.shape[0], META['BLOCK_SIZE']),)

    # Launch the kernel
    combined_activation_kernel[grid](
        input.data.ptr,
        weight1.data.ptr,
        weight2.data.ptr,
        bias.data.ptr,
        out.data.ptr,
        input.shape[0],
        input.shape[-1],
        weight1.shape[1],
    )

    return out
