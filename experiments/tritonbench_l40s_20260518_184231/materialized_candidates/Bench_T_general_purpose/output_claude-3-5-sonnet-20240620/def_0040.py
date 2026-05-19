{{ code }}
import triton
import triton.language as tl

@triton.jit
def sigmoid_batch_norm_kernel(input_ptr, running_mean_ptr, running_var_ptr, weight_ptr, bias_ptr, output_ptr, N, C, momentum, eps):
    # Define the grid size
    pid = tl.program_id(0)
    # Calculate the index for the input tensor
    idx = pid * tl.num_warps() + tl.arange(0, tl.num_warps())
    
    # Load input tensor
    input_val = tl.load(input_ptr + idx)
    
    # Load running mean and variance
    running_mean = tl.load(running_mean_ptr)
    running_var = tl.load(running_var_ptr)
    
    # Normalize the input
    normalized_input = (input_val - running_mean) / tl.sqrt(running_var + eps)
    
    # Apply weight and bias if provided
    if weight_ptr is not None:
        weight = tl.load(weight_ptr)
        normalized_input *= weight
    if bias_ptr is not None:
        bias = tl.load(bias_ptr)
        normalized_input += bias
    
    # Apply sigmoid activation function
    output_val = 1 / (1 + tl.exp(-normalized_input))
    
    # Store the output
    tl.store(output_ptr + idx, output_val)

def sigmoid_batch_norm(input, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-5):
    # Get the shape of the input tensor
    N, C = input.shape[0], input.shape[1]
    
    # Allocate output tensor
    output = torch.empty_like(input)
    
    # Launch the kernel
    sigmoid_batch_norm_kernel[(N,)](input, running_mean, running_var, weight, bias, output, N, C, momentum, eps)
    
    return output
{{ code }}
