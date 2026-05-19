import triton
import triton.language as tl

@triton.jit
def trunc_kernel(input_ptr, output_ptr, n_elements, **meta):
    # Compute the index of the element this program instance will process
    pid = tl.program_id(0)
    # Load the input element
    input_val = tl.load(input_ptr + pid)
    # Apply truncation operation
    output_val = tl.math.trunc(input_val)
    # Store the result in the output tensor
    tl.store(output_ptr + pid, output_val)

def trunc(input, *, out=None):
    # Ensure the input is a tensor
    assert isinstance(input, triton.Tensor), "Input must be a Triton tensor"

    # Create an output tensor if not provided
    if out is None:
        out = triton.empty_like(input)

    # Ensure the output tensor is a Triton tensor
    assert isinstance(out, triton.Tensor), "Output must be a Triton tensor"

    # Get the number of elements in the input tensor
    n_elements = input.numel()

    # Launch the Triton kernel
    grid = (n_elements,)
    trunc_kernel[grid](input, out, n_elements)

    return out
