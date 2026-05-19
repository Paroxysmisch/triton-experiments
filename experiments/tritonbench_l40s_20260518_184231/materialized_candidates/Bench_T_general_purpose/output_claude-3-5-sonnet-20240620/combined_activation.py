import triton
import triton.language as tl

@triton.jit
def combined_activation_kernel(input_ptr, weight1_ptr, weight2_ptr, bias_ptr, output_ptr, N, D_in, D_out, batch_size):
    # Calculate the batch index
    batch_idx = tl.program_id(0)
    
    # Allocate space for intermediate results
    X = tl.load(input_ptr + batch_idx * N * D_in)
    W1 = tl.load(weight1_ptr)
    
    # Matrix multiplication: X @ W1
    Z = tl.dot(X, W1)  # Shape: (N, D_out)
    
    # Apply sigmoid
    sigmoid_Z = 1 / (1 + tl.exp(-Z))
    
    # Apply tanh
    tanh_sigmoid_Z = (tl.exp(sigmoid_Z) - tl.exp(-sigmoid_Z)) / (tl.exp(sigmoid_Z) + tl.exp(-sigmoid_Z))
    
    # Load weight2 and bias
    W2 = tl.load(weight2_ptr)
    b = tl.load(bias_ptr)
    
    # Element-wise multiplication and addition
    output = (tanh_sigmoid_Z * W2) + b
    
    # Store the result
    tl.store(output_ptr + batch_idx * N * D_out, output)

def combined_activation(input: Tensor, weight1: Tensor, weight2: Tensor, bias: Tensor, *, out: Optional[Tensor] = None) -> Tensor:
    # Ensure input dimensions are compatible
    assert input.dim() >= 2, "Input must have at least 2 dimensions."
    assert weight1.shape[0] == input.shape[-1], "Weight1 dimensions must match input features."
    assert weight2.shape == (1,) or weight2.shape == input.shape[:-1] + weight1.shape[1:], "Weight2 must be broadcastable to the output shape."
    assert bias.shape == (1,) or bias.shape == input.shape[:-1] + (weight1.shape[1],), "Bias must be broadcastable to the output shape."
    
    # Get dimensions
    batch_size, N, D_in = input.shape
    D_out = weight1.shape[1]
    
    # Allocate output tensor
    if out is None:
        out = input.new_empty((batch_size, N, D_out))
    
    # Launch the Triton kernel
    grid = (batch_size,)
    combined_activation_kernel[grid](input, weight1, weight2, bias, out, N, D_in, D_out, batch_size)
    
    return out
