import triton
import triton.language as tl

@triton.jit
def tanh_kernel(input_ptr, output_ptr, n_elements):
    # Compute the hyperbolic tangent for each element
    pid = tl.program_id(0)
    # Calculate the index for the current thread
    index = pid * tl.num_warps() + tl.arange(0, tl.num_warps())
    # Ensure we do not go out of bounds
    mask = index < n_elements
    # Load input tensor elements
    input_val = tl.load(input_ptr + index, mask=mask)
    # Compute hyperbolic tangent
    output_val = tl.tanh(input_val)
    # Store the result in the output tensor
    tl.store(output_ptr + index, output_val, mask=mask)

def tanh(input, *, out=None):
    # Get the number of elements in the input tensor
    n_elements = input.numel()
    # Allocate output tensor if not provided
    if out is None:
        out = input.new_zeros(input.shape, dtype=input.dtype)
    # Launch the Triton kernel
    tanh_kernel[(n_elements + 255) // 256](input.data_ptr(), out.data_ptr(), n_elements)
    return out
