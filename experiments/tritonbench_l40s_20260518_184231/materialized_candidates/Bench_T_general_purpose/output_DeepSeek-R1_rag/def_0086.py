import torch
import triton
import triton.language as tl

@triton.jit
def log_tanh_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input_values = tl.load(input_ptr + offsets, mask=mask)
    log_values = tl.log(input_values)
    tanh_values = tl.tanh(log_values)
    tl.store(output_ptr + offsets, tanh_values, mask=mask)

def log_tanh(input, out=None):
    assert input.is_cuda, "Input tensor must be on CUDA"
    if not torch.all(input > 0):
        raise ValueError("All elements in the input tensor must be positive.")
    if out is None:
        out = torch.empty_like(input)
    else:
        assert out.shape == input.shape, "Output tensor must have the same shape as input"
        assert out.is_cuda, "Output tensor must be on CUDA"
    n_elements = input.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    log_tanh_kernel[grid](input, out, n_elements, BLOCK_SIZE=1024)
    return out

# Test the function
torch.manual_seed(0)
size = 98432
x = torch.rand(size, device='cuda')  # All elements are positive
output_triton = log_tanh(x)
output_torch = torch.tanh(torch.log(x))

print(f"The maximum difference between torch and triton is "
      f"{torch.max(torch.abs(output_torch - output_triton))}")
