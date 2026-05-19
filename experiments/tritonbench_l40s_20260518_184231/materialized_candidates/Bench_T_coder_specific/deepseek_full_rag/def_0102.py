import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(input_ptr, other_ptr, output_ptr, n_elements,
                   stride_input_batch, stride_input_last,
                   stride_output_batch, stride_output_last,
                   BLOCK_SIZE: tl.constexpr):
    # Compute the program ID
    pid_batch = tl.program_id(0)
    pid_last = tl.program_id(1)

    # Compute the start pointer for this program
    input_ptr += pid_batch * stride_input_batch + pid_last * stride_input_last
    other_ptr += pid_batch * stride_input_batch + pid_last * stride_input_last
    output_ptr += pid_batch * stride_output_batch + pid_last * stride_output_last

    # Create offsets for the block
    offsets = tl.arange(0, BLOCK_SIZE)

    # Load input and other tensors
    input_local = tl.load(input_ptr + offsets, mask=offsets < n_elements, other=0.0)
    other_local = tl.load(other_ptr + offsets, mask=offsets < n_elements, other=0.0)

    # Compute the maximum value
    input_local_minus_max = input_local - tl.max(input_local, axis=0)

    # Compute the exponent
    exp_local = tl.exp(input_local_minus_max)

    # Compute the sum
    sum_local = tl.sum(exp_local, axis=0)

    # Compute the softmax
    softmax_local = exp_local / sum_local

    # Multiply the softmax by other
    output_local = softmax_local * other_local

    # Store the result
    tl.store(output_ptr + offsets, output_local, mask=offsets < n_elements)

def softmax_mul(input, other, dim=-1, dtype=None, out=None):
    # Handle dtype and out arguments
    if dtype is not None:
        input = input.to(dtype)
    if out is not None:
        out = out.to(input.dtype)

    # Check that dim is valid
    if dim < -input.ndim or dim >= input.ndim:
        raise IndexError(
            "Dimension out of range (expected to be in range of [{}, {}], but got {})".format(
                -input.ndim, input.ndim - 1, dim))

    # Ensure input and other have the same shape
    if input.shape != other.shape:
        raise ValueError("Input and other must have the same shape")

    # Reshape input and other for processing
    input_reshaped = input.reshape(-1, input.shape[-1])
    other_reshaped = other.reshape(-1, other.shape[-1])

    batch_size, n_elements = input_reshaped.shape

    # Allocate output tensor
    output = torch.empty_like(input_reshaped, dtype=torch.float32)

    # Define the grid
    grid = lambda meta: (triton.cdiv(batch_size, meta['BLOCK_SIZE']), n_elements)

    # Call the Triton kernel
    softmax_kernel[grid](input_reshaped, other_reshaped, output, n_elements,
                         input_reshaped.stride(0), input_reshaped.stride(1),
                         output.stride(0), output.stride(1))

    return output.to(input.dtype)
