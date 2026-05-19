import triton
import triton.language as tl
from triton.language.math import erf, pow, tanh

# Kernel to compute GELU using the error function approximation
@triton.jit
def gelu_none_kernel(x, out):
    x_fp32 = x.to(tl.float32)
    x_gelu = 0.5 * x_fp32 * (1 + erf(x_fp32 * 0.7071067811))
    out[tl.program_id(0)] = x_gelu

# Kernel to compute GELU using the tanh approximation
@triton.jit
def gelu_tanh_kernel(x, out):
    x_fp32 = x.to(tl.float32)
    x_gelu = 0.5 * x_fp32 * (1 + tanh(x_fp32 * 0.79788456 * (1 + 0.044715 * pow(x_fp32.to(tl.float32), 2))))
    out[tl.program_id(0)] = x_gelu

# Reduction kernel to find the minimum value
@triton.jit
def reduce_min_kernel(in_ptr, out_ptr, n_elements):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    active_in = tl.load(in_ptr + block_start + offsets, mask=mask)
    min_val = tl.min(active_in)
    tl.store(out_ptr + pid, min_val)

BLOCK_SIZE = 1024

def min_gelu(input, dim=None, keepdim=False, approximate='none', out=None):
    if approximate not in ['none', 'tanh']:
        raise ValueError(f"Invalid approximate value: {approximate}")
    
    input_shape = input.shape
    num_elements = input.numel()
    
    # Allocate memory for intermediate results
    if approximate == 'none':
        gelu_out = input.new_empty_like(input)
        grid_size = (num_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
        gelu_none_kernel[grid_size, BLOCK_SIZE](input, gelu_out)
    else:
        gelu_out = input.new_empty_like(input)
        grid_size = (num_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
        gelu_tanh_kernel[grid_size, BLOCK_SIZE](input, gelu_out)
    
    if dim is None:
        reduced_output = gelu_out.new_empty(())
        reduce_min_kernel[gelu_out.size(0), BLOCK_SIZE](gelu_out, reduced_output, num_elements)
        return reduced_output
    
    # Reduce along the specified dimension
    reduced_shape = list(input_shape)
    reduced_shape[dim] = 1 if keepdim else 0
    reduced_output = gelu_out.new_empty(reduced_shape)
    reduce_min_kernel[reduced_output.size(0), BLOCK_SIZE](gelu_out, reduced_output, num_elements)
    
    return reduced_output

# Example usage
if __name__ == "__main__":
    import torch
    input_tensor = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float32)
    result = min_gelu(input_tensor, dim=None, approximate='tanh')
    print(result.item())  # Output should be the minimum GELU value
