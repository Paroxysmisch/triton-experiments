import torch
import triton
import triton.language as tl

@triton.jit
def _permute_copy(input, dims):
    # Get the shape of the input tensor
    shape = input.shape
    # Create a new tuple of dimensions based on the 'dims' parameter
    new_dims = tuple(shape[i] for i in dims)
    # Reshape the input tensor to the new dimensions
    return input.reshape(new_dims)

def torch_permute_copy(input, dims):
    # Call the Triton kernel _permute_copy
    return _permute_copy(input, dims)

# Test the wrapper function with provided inputs
def test_wrapper(func_name, *, num_args, tu_options=None):
    args = [rand_strided((3, 4, 5), s32) for _ in range(num_args)]
    out_torch = getattr(torch, func_name)(*args)
    out_triton = eval(func_name)(*args)
    assert_close(out_torch, out_triton)
