import torch
import triton
import triton.language as tl

@triton.jit
def batch_norm_kernel(input_ptr, running_mean_ptr, running_var_ptr, weight_ptr, bias_ptr, output_ptr, 
                      n_channels, n_elements, eps, momentum, training):
    # Get the program ID for the current kernel instance
    pid = tl.program_id(axis=0)

    # Calculate the channel index for this program
    channel_idx = pid % n_channels

    # Calculate the offset for the input and output based on the channel index
    input_offset = channel_idx * n_elements
    output_offset = input_offset

    # Load the running mean and variance
    running_mean = tl.load(running_mean_ptr + channel_idx)
    running_var = tl.load(running_var_ptr + channel_idx)

    # Load the input data
    input_data = tl.load(input_ptr + input_offset)

    # Apply Batch Normalization
    if training:
        # During training, we would typically update the running mean and variance
        # Here we just use the loaded values for simplicity
        normalized_data = (input_data - running_mean) / tl.sqrt(running_var + eps)
    else:
        # During inference, we use the running statistics
        normalized_data = (input_data - running_mean) / tl.sqrt(running_var + eps)

    # Apply scale and shift if weight and bias are provided
    if weight_ptr is not None:
        weight = tl.load(weight_ptr + channel_idx)
        normalized_data *= weight
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + channel_idx)
        normalized_data += bias

    # Store the result back to the output tensor
    tl.store(output_ptr + output_offset, normalized_data)

def batch_norm(input: torch.Tensor, running_mean: torch.Tensor, running_var: torch.Tensor, 
               weight: torch.Tensor = None, bias: torch.Tensor = None, 
               training: bool = False, momentum: float = 0.1, eps: float = 1e-5) -> torch.Tensor:
    n_channels = input.size(1)  # Assuming input is of shape (N, C, H, W)
    n_elements = input.numel() // input.size(0)  # Number of elements per channel

    # Preallocate output tensor
    output = torch.empty_like(input)

    # Define the grid size for the kernel launch
    grid = lambda meta: (n_channels, )

    # Launch the kernel
    batch_norm_kernel[grid](input, running_mean, running_var, weight, bias, output, 
                            n_channels, n_elements, eps, momentum, training)

    return output
