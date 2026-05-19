import triton
import triton.language as tl

@triton.jit
def softplus_linear_kernel(input_ptr, weight_ptr, bias_ptr, output_ptr, N, beta, threshold):
    # Get the index of the current element
    idx = tl.program_id(0) * tl.block_size(0) + tl.arange(0, tl.block_size(0))
    mask = idx < N  # Ensure we don't go out of bounds

    # Load input, weight, and bias
    input_val = tl.load(input_ptr + idx, mask=mask)
    weight_val = tl.load(weight_ptr)
    bias_val = tl.load(bias_ptr) if bias_ptr is not None else 0.0

    # Linear transformation
    linear_out = input_val * weight_val + bias_val

    # Softplus activation
    softplus_out = (1 / beta) * tl.log(1 + tl.exp(beta * linear_out))

    # Apply threshold for numerical stability
    output_val = tl.where(linear_out > threshold, linear_out, softplus_out)

    # Store the result
    tl.store(output_ptr + idx, output_val, mask=mask)

def softplus_linear(input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None, beta: float = 1.0, threshold: float = 20.0) -> torch.Tensor:
    # Ensure input dimensions are compatible
    assert input.dim() == 1, "Input tensor must be 1-dimensional"
    assert weight.dim() == 1 and weight.size(0) == input.size(0), "Weight tensor must be 1-dimensional and match input size"
    if bias is not None:
        assert bias.dim() == 1 and bias.size(0) == 1, "Bias tensor must be 1-dimensional and of size 1"

    # Prepare output tensor
    output = torch.empty_like(input)

    # Launch the Triton kernel
    grid = (input.size(0) + 255) // 256  # Adjust block size as needed
    softplus_linear_kernel[grid](input.data_ptr(), weight.data_ptr(), bias.data_ptr() if bias is not None else None, output.data_ptr(), input.size(0), beta, threshold)

    return output
