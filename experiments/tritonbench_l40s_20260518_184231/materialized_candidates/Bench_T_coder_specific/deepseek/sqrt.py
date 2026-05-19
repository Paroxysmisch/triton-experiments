import triton
import triton.language as tl

@triton.jit
def _sqrt_kernel(input_ptr, output_ptr, n_elements):
    # Get the index of the current element
    idx = tl.program_id()
    # Check if the current index is within the bounds of the array
    if idx < n_elements:
        # Load the value of the current element from the input array
        value = tl.load(input_ptr + idx)
        # Compute the square root of the value
        result = tl.sqrt(value)
        # Store the result in the output array
        tl.store(output_ptr + idx, result)

def sqrt(input_tensor, out=None):
    # Check if the input tensor is a Tensor object
    if not isinstance(input_tensor, triton.Tensor):
        raise ValueError("input_tensor must be a Tensor object")
    # Check if the output tensor is a Tensor object or None
    if out is not None and not isinstance(out, triton.Tensor):
        raise ValueError("out must be a Tensor object or None")
    # If the output tensor is None, create a new tensor for the output
    if out is None:
        out = triton.Tensor(input_tensor.shape, input_tensor.dtype)
    # Get the number of elements in the input tensor
    n_elements = input_tensor.numel()
    # Run the kernel
    _sqrt_kernel[n_elements,](input_tensor.device_buffer, out.device_buffer, n_elements)
    return out
