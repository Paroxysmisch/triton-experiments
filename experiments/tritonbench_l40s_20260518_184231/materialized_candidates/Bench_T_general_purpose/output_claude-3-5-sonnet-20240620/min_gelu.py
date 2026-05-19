# triton_kernel.py
import triton
import triton.language as tl
from collections import namedtuple

@triton.jit
def gelu_kernel(input_ptr, output_ptr, n_elements, approximate):
    # Compute GELU activation
    idx = tl.program_id(0)
    if idx < n_elements:
        x = input_ptr[idx]
        if approximate == 0:  # 'none'
            # Exact GELU
            output_ptr[idx] = x * 0.5 * (1 + tl.erf(x / tl.sqrt(2.0)))
        elif approximate == 1:  # 'tanh'
            # Approximate GELU using tanh
            output_ptr[idx] = 0.5 * x * (1 + tl.tanh(tl.sqrt(2 / tl.pi) * (x + 0.044715 * x**3)))

def min_gelu(input, dim=None, keepdim=False, approximate='none', out=None):
    # Convert approximate string to integer
    approx_map = {'none': 0, 'tanh': 1}
    approx = approx_map.get(approximate, 0)

    # Prepare output tensor
    if out is None:
        out = input.new_zeros(input.shape)

    # Launch GELU kernel
    n_elements = input.numel()
    gelu_output = input.new_zeros_like(input)
    gelu_kernel[(n_elements,)](input, gelu_output, n_elements, approx)

    # Compute minimum
    if dim is not None:
        values, indices = tl.min(gelu_output, dim=dim, keepdim=keepdim)
        return namedtuple('MinGELU', ['values', 'indices'])(values, indices)
    else:
        return tl.min(gelu_output)
