import torch
import triton
import triton.language as tl

@triton.jit
def softplus_linear_kernel(output_ptr, input_ptr, weight_ptr, bias_ptr, n_elements, beta, threshold, BLOCK_SIZE: tl.constexpr):
    # Compute the index for the current element
    idx = tl.program_id(0)
    
    # Load input, weight, and bias
    input_val = tl.load(input_ptr + idx)
    weight_val = tl.load(weight_ptr)
    bias_val = tl.load(bias_ptr) if bias_ptr is not None else 0.0
    
    # Perform linear transformation
    linear_output = input_val * weight_val + bias_val
    
    # Apply Softplus activation
    # For numerical stability, we check against the threshold
    softplus_output = tl.where(linear_output > threshold, linear_output, (1 / beta) * tl.log(1 + tl.exp(beta * linear_output)))
    
    # Store the result
    tl.store(output_ptr + idx, softplus_output)

def softplus_linear(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None, beta: float = 1, threshold: float = 20) -> torch.Tensor:
    n_elements = input.numel()
    # Allocate output tensor
    output = torch.empty_like(input)
    
    # Define block size
    BLOCK_SIZE = 256  # You can adjust this based on your needs
    
    # Enqueue the kernel
    softplus_linear_kernel[(n_elements,)](
        output,
        input,
        weight,
        bias,
        n_elements,
        beta,
        threshold,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output
