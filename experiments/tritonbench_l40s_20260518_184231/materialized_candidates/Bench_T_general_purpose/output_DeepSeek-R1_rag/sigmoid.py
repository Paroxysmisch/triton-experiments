import torch
import triton
import triton.language as tl

@triton.jit
def sigmoid_kernel(
    input_ptr,  # Pointer to the input tensor.
    output_ptr,  # Pointer to the output tensor.
    n_elements,  # Number of elements in the tensor.
    BLOCK_SIZE: tl.constexpr,  # Number of elements processed per program.
):
    pid = tl.program_id(axis=0)  # 1D launch grid.
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input elements.
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    # Compute sigmoid.
    output = 1.0 / (1.0 + tl.exp(-x))
    # Store output.
    tl.store(output_ptr + offsets, output, mask=mask)


def sigmoid(input: torch.Tensor, *, out=None) -> torch.Tensor:
    # Check input is on CUDA.
    if not input.is_cuda:
        raise TypeError("Input tensor must be a CUDA tensor.")
    
    # Ensure input is contiguous.
    if not input.is_contiguous():
        input = input.contiguous()
    
    # Validate or create output tensor.
    if out is not None:
        if not out.is_cuda:
            raise TypeError("Output tensor must be a CUDA tensor.")
        if out.dtype != input.dtype:
            raise TypeError(f"Output tensor dtype ({out.dtype}) must match input dtype ({input.dtype}).")
        if out.shape != input.shape:
            raise RuntimeError(f"Output tensor shape {out.shape} does not match input shape {input.shape}.")
        if not out.is_contiguous():
            raise RuntimeError("Output tensor must be contiguous.")
        output = out
    else:
        output = torch.empty_like(input)
    
    n_elements = input.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # Launch kernel with default BLOCK_SIZE.
    sigmoid_kernel[grid](input, output, n_elements, BLOCK_SIZE=1024)
    
    return output

# Verify the implementation with a test.
torch.manual_seed(0)
size = 98432
x = torch.rand(size, device='cuda')
output_torch = torch.special.expit(x)
output_triton = sigmoid(x)
print(f"Maximum difference: {torch.max(torch.abs(output_torch - output_triton))}")
