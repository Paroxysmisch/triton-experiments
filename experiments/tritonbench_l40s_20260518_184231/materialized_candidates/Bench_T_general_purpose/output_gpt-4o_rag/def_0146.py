import torch
import triton
import triton.language as tl

# Triton kernel for combined Linear and ELU activation
@triton.jit
def elu_linear_kernel(input_ptr, weight_ptr, bias_ptr, output_ptr, alpha_ptr,
                      n_elements, n_features, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_elements

    # Load input and weights
    input = tl.load(input_ptr + offsets, mask=mask)
    weight = tl.load(weight_ptr + tl.arange(0, n_features))

    # Perform linear transformation
    linear_output = tl.dot(input, weight)
    if bias_ptr:
        bias = tl.load(bias_ptr + offsets, mask=mask)
        linear_output += bias

    # Load alpha
    alpha = tl.load(alpha_ptr)

    # Apply ELU activation
    elu_output = tl.where(linear_output > 0, linear_output, alpha * (tl.exp(linear_output) - 1))

    # Store result
    tl.store(output_ptr + offsets, elu_output, mask=mask)

# Function to call the Triton kernel
def elu_linear(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None,
               alpha: float = 1.0, inplace: bool = False) -> torch.Tensor:
    assert input.is_cuda and weight.is_cuda
    if bias is not None:
        assert bias.is_cuda

    n_elements = input.numel()
    n_features = weight.size(0)

    output = input if inplace else torch.empty_like(input)

    # Prepare alpha as a tensor
    alpha_tensor = torch.tensor([alpha], device=input.device)

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    elu_linear_kernel[grid](input, weight, bias, output, alpha_tensor, n_elements, n_features, BLOCK_SIZE=1024)

    return output

# Example usage
torch.manual_seed(0)
input_size = (1024, 512)
input = torch.rand(input_size, device='cuda')
weight = torch.rand((512,), device='cuda')
bias = torch.rand((1024,), device='cuda')
alpha = 1.0

output = elu_linear(input, weight, bias, alpha, inplace=False)
print(output)
