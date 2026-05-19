import triton
import triton.language as tl

@triton.jit
def update_fn_kernel(
    p_ptr,  # Pointer to the parameters
    grad_ptr,  # Pointer to the gradients
    exp_avg_ptr,  # Pointer to the exponential moving averages
    n_elements,  # Number of elements
    weight_decay,  # Weight decay factor
    momentum,  # Momentum factor
    learning_rate,  # Learning rate
    BLOCK_SIZE: tl.constexpr  # Block size
):
    pid = tl.program_id(axis=0)  # Get the program ID
    block_start = pid * BLOCK_SIZE  # Start index of the block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)  # Offsets within the block

    # Load data from global memory
    p = tl.load(p_ptr + offsets, mask=offsets < n_elements)
    grad = tl.load(grad_ptr + offsets, mask=offsets < n_elements)
    exp_avg = tl.load(exp_avg_ptr + offsets, mask=offsets < n_elements)

    # Apply weight decay
    p = p * (1 - weight_decay * learning_rate)

    # Update exponential moving average
    exp_avg = momentum * exp_avg + (1 - momentum) * grad

    # Update parameters
    p = p - learning_rate * exp_avg

    # Store updated data back to global memory
    tl.store(p_ptr + offsets, p, mask=offsets < n_elements)
    tl.store(exp_avg_ptr + offsets, exp_avg, mask=offsets < n_elements)

import torch

def update_fn(p, grad, exp_avg, weight_decay, momentum, learning_rate):
    # Ensure the tensors are on the same device
    assert p.device == grad.device == exp_avg.device, "All tensors must be on the same device"
    device = p.device

    # Convert tensors to pointers
    p_ptr = p.data_ptr()
    grad_ptr = grad.data_ptr()
    exp_avg_ptr = exp_avg.data_ptr()

    # Number of elements
    n_elements = p.numel()

    # Define block size
    BLOCK_SIZE = 256

    # Determine grid size
    grid_size = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch the kernel
    update_fn_kernel[grid_size, BLOCK_SIZE](
        p_ptr, grad_ptr, exp_avg_ptr, n_elements, weight_decay, momentum, learning_rate
    )

# Example usage
if __name__ == "__main__":
    # Create example tensors
    p = torch.tensor([1.0, 2.0, 3.0, 4.0], device='cuda')
    grad = torch.tensor([0.1, 0.2, 0.3, 0.4], device='cuda')
    exp_avg = torch.tensor([0.0, 0.0, 0.0, 0.0], device='cuda')

    # Parameters
    weight_decay = 0.01
    momentum = 0.9
    learning_rate = 0.01

    # Perform the update
    update_fn(p, grad, exp_avg, weight_decay, momentum, learning_rate)

    # Print the updated parameters and exponential moving averages
    print("Updated parameters:", p)
    print("Updated exponential moving averages:", exp_avg)
