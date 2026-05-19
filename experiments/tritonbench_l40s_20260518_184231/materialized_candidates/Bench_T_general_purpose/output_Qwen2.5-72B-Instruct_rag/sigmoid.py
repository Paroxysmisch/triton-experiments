import torch
import triton
import triton.language as tl

# Triton kernel for computing the sigmoid function
@triton.jit
def sigmoid_kernel(x_ptr,  # *Pointer* to first input vector.
                   output_ptr,  # *Pointer* to output vector.
                   n_elements,  # Size of the vector.
                   BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
                   # NOTE: `constexpr` so it can be used as a shape value.
                   ):

    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.

    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)

    output = 1 / (1 + tl.exp(-x))

    tl.store(output_ptr + offsets, output, mask=mask)

# Wrapper function for the sigmoid kernel
def sigmoid(input: torch.Tensor, *, out: torch.Tensor = None) -> torch.Tensor:
    if out is None:
        out = torch.empty_like(input)
    else:
        assert out.shape == input.shape, "Output tensor must have the same shape as input tensor"
        assert out.dtype == input.dtype, "Output tensor must have the same data type as input tensor"
        assert out.is_cuda, "Output tensor must be on the same device as input tensor"

    assert input.is_cuda, "Input tensor must be on a CUDA device"
    n_elements = input.numel()

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )

    sigmoid_kernel[grid](input, out, n_elements, BLOCK_SIZE=1024)

    return out

# Example usage
torch.manual_seed(0)
size = 98432
x = torch.rand(size, device='cuda')
output_torch = torch.sigmoid(x)
output_triton = sigmoid(x)

print(output_torch)
print(output_triton)
print(f'The maximum difference between torch and triton is '
      f'{torch.max(torch.abs(output_torch - output_triton))}')
