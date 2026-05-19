import torch
import triton
import triton.language as tl

# Triton kernel for linear transformation followed by ELU activation
@triton.jit
def elu_linear_kernel(input_ptr, weight_ptr, bias_ptr, alpha_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_elements

    # Load input, weight, and bias
    input_tensor = tl.load(input_ptr + offsets, mask=mask)
    weight_tensor = tl.load(weight_ptr)
    bias_tensor = tl.load(bias_ptr) if bias_ptr is not None else 0.0

    # Linear transformation
    linear_output = tl.dot(input_tensor, weight_tensor) + bias_tensor

    # Load alpha for ELU
    alpha = tl.load(alpha_ptr)

    # Apply ELU activation
    output = tl.where(linear_output > 0, linear_output, alpha * (tl.exp(linear_output) - 1))

    # Store the result
    tl.store(output_ptr + offsets, output, mask=mask)

# Wrapper function for the ELU linear transformation
def elu_linear(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None, alpha: float = 1.0, inplace: bool = False) -> torch.Tensor:
    output = torch.empty_like(input)
    assert input.is_cuda and output.is_cuda
    n_elements = input.numel()

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )

    # Prepare bias pointer
    bias_ptr = None if bias is None else bias.data_ptr()

    # Call the Triton kernel
    elu_linear_kernel[grid](input, weight, bias_ptr, alpha, output, n_elements, BLOCK_SIZE=1024)

    return output

# Example usage
torch.manual_seed(0)
size = 98432
input_tensor = torch.rand(size, device='cuda') * 2 - 1
weight_tensor = torch.rand(size, device='cuda')
bias_tensor = torch.rand(1, device='cuda')
alpha = torch.tensor(1.0, device='cuda')

output_torch = torch.where(input_tensor > 0, input_tensor, alpha * (torch.exp(input_tensor) - 1))
output_triton = elu_linear(input_tensor, weight_tensor, bias=bias_tensor, alpha=alpha.item())

print(f'The maximum difference between torch and triton is {torch.max(torch.abs(output_torch - output_triton))}')
