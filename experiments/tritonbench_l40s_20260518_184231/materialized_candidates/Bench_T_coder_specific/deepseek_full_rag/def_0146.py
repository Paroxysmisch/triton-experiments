import torch
import triton
import triton.language as tl

# Triton kernel for ELU activation function
@triton.jit
def elu_kernel(x_ptr,  # *Pointer* to first input vector.
               alpha_ptr,
               output_ptr,  # *Pointer* to output vector.
               n_elements,  # Size of the vector.
               BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
               ):

    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.

    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < n_elements
  
    x = tl.load(x_ptr + offsets, mask=mask)
    alpha = tl.load(alpha_ptr)

    output = tl.where(x > 0, x, alpha * (tl.exp(x) - 1))

    tl.store(output_ptr + offsets, output, mask=mask)

# Function to call the Triton kernel
def elu(x: torch.Tensor, alpha: torch.Tensor):
    output = torch.empty_like(x)
    assert x.is_cuda and output.is_cuda
    n_elements = output.numel()

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )

    elu_kernel[grid](x, alpha, output, n_elements, BLOCK_SIZE=1024)

    return output

# Example usage
torch.manual_seed(0)
size = 98432
x = torch.rand(size, device='cuda') * 2 - 1
alpha = torch.rand(1, device='cuda')
output_torch = torch.where(x > 0, x, alpha * (torch.exp(x) - 1))
output_triton = elu(x, alpha)
print(alpha)
print(x)
print(output_torch)
print(output_triton)
print(f'The maximum difference between torch and triton is '
      f'{torch.max(torch.abs(output_torch - output_triton))}')
