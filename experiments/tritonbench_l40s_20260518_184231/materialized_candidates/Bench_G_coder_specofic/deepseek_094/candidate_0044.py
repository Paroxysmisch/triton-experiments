import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
    input_ptr,
    output_ptr,
    row_stride,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    input_offset = row * row_stride
    output_offset = row * row_stride

    # Load data into SRAM
    row_data = tl.load(input_ptr + input_offset, BLOCK_SIZE)

    # Subtract maximum value for numerical stability
    max_val = tl.max(row_data, axis=0)
    row_data = row_data - max_val

    # Apply exponential function
    row_data = tl.exp(row_data)

    # Compute denominator
    denom = tl.sum(row_data, axis=0)

    # Compute output
    output_data = row_data / denom

    # Store back to memory
    tl.store(output_ptr + output_offset, output_data)


def softmax(input_tensor):
    n_rows, n_cols = input_tensor.shape
    BLOCK_SIZE = 2**(tl.math.ceil(tl.log2(n_cols)))

    # Create output tensor
    output_tensor = tl.empty_like(input_tensor)

    # Invoke kernel
    grid = lambda meta: (n_rows, )
    softmax_kernel[grid](input_tensor, output_tensor, n_cols, BLOCK_SIZE)

    return output_tensor
