import torch
import triton
import triton.language as tl

# Triton kernel for linear transformation followed by ELU activation
@triton.jit
def elu_linear_kernel(
    input_ptr,  # *Pointer* to input tensor.
    weight_ptr,  # *Pointer* to weight tensor.
    bias_ptr,  # *Pointer* to bias tensor (optional).
    alpha_ptr,  # *Pointer* to alpha parameter.
    output_ptr,  # *Pointer* to output tensor.
    input_size,  # Number of elements in the input tensor.
    output_size,  # Number of elements in the output tensor.
    input_stride,  # Stride of the input tensor.
    weight_stride,  # Stride of the weight tensor.
    bias_stride,  # Stride of the bias tensor (optional).
    BLOCK_SIZE_INPUT: tl.constexpr,  # Number of elements each program should process for input.
    BLOCK_SIZE_OUTPUT: tl.constexpr  # Number of elements each program should process for output.
):
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.

    block_start_input = pid * BLOCK_SIZE_INPUT
    block_start_output = pid * BLOCK_SIZE_OUTPUT
    offsets_input = block_start_input + tl.arange(0, BLOCK_SIZE_INPUT)
    offsets_output = block_start_output + tl.arange(0, BLOCK_SIZE_OUTPUT)

    mask_input = offsets_input < input_size
    mask_output = offsets_output < output_size

    input = tl.load(input_ptr + offsets_input * input_stride, mask=mask_input)
    weight = tl.load(weight_ptr + offsets_output * weight_stride, mask=mask_output)

    # Perform the linear transformation
    output = tl.zeros((BLOCK_SIZE_OUTPUT,), dtype=tl.float32)
    for i in range(input_size):
        output += input[i] * weight[i]

    # Add bias if provided
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + offsets_output * bias_stride, mask=mask_output)
        output += bias

    # Apply ELU activation
    alpha = tl.load(alpha_ptr)
    output = tl.where(output > 0, output, alpha * (tl.exp(output) - 1))

    tl.store(output_ptr + offsets_output, output, mask=mask_output)

# Function to call the Triton kernel
def elu_linear(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None, alpha: float = 1.0, inplace: bool = False) -> torch.Tensor:
    if inplace:
        output = input
    else:
        output = torch.empty((input.size(0), weight.size(0)), device=input.device, dtype=input.dtype)

    assert input.is_cuda and weight.is_cuda and output.is_cuda
    if bias is not None:
        assert bias.is_cuda

    input_size = input.numel()
    output_size = output.numel()

    grid = lambda meta: (triton.cdiv(output_size, meta['BLOCK_SIZE_OUTPUT']), )

    elu_linear_kernel[grid](
        input, weight, bias, torch.tensor([alpha], device=input.device, dtype=input.dtype),
        output, input_size, output_size, input.stride(0), weight.stride(0), bias.stride(0) if bias is not None else 0,
        BLOCK_SIZE_INPUT=1024, BLOCK_SIZE_OUTPUT=1024
    )

    return output

# Example usage
torch.manual_seed(0)
input_size = 128
output_size = 64
batch_size = 32

input = torch.rand(batch_size, input_size, device='cuda')
weight = torch.rand(output_size, input_size, device='cuda')
bias = torch.rand(output_size, device='cuda')
alpha = 1.0

output_triton = elu_linear(input, weight, bias, alpha, inplace=False)
print(output_triton)
