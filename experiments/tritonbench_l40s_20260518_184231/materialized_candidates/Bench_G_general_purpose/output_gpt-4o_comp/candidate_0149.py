import triton
import triton.language as tl
import torch

# Define the Triton kernel
@triton.jit
def update_fn_kernel(
    p_ptr, grad_ptr, exp_avg_ptr,
    lr, wd, beta1, beta2,
    n_elements, BLOCK_SIZE: tl.constexpr
):
    # Compute the program ID
    pid = tl.program_id(axis=0)

    # Compute the start index for this block
    start = pid * BLOCK_SIZE

    # Create an offset for this block
    offsets = start + tl.arange(0, BLOCK_SIZE)

    # Mask to ensure we only process valid elements
    mask = offsets < n_elements

    # Load the parameters, gradients, and exponential moving averages
    p = tl.load(p_ptr + offsets, mask=mask)
    grad = tl.load(grad_ptr + offsets, mask=mask)
    exp_avg = tl.load(exp_avg_ptr + offsets, mask=mask)

    # Step weight decay
    p = p * (1 - lr * wd)

    # Calculate the difference between exp_avg and grad
    delta = exp_avg - grad

    # Update parameters with momentum
    update = beta1 * delta
    p = p + update

    # Apply sign-based adjustment
    sign_adjustment = tl.where(delta != 0, tl.sign(delta), 0)
    p = p + sign_adjustment

    # Update exponential moving average
    exp_avg = beta2 * exp_avg + (1 - beta2) * grad

    # Store the updated parameters and exponential averages
    tl.store(p_ptr + offsets, p, mask=mask)
    tl.store(exp_avg_ptr + offsets, exp_avg, mask=mask)

# Define the wrapper function
def update_fn(p, grad, exp_avg, lr, wd, beta1, beta2, BLOCK_SIZE=1024):
    # Ensure all tensors are CUDA tensors
    assert p.is_cuda and grad.is_cuda and exp_avg.is_cuda

    # Calculate the number of elements
    n_elements = p.numel()

    # Calculate the number of blocks
    grid = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch the Triton kernel
    update_fn_kernel[grid](
        p, grad, exp_avg,
        lr, wd, beta1, beta2,
        n_elements, BLOCK_SIZE=BLOCK_SIZE
    )

# Example usage
if __name__ == "__main__":
    # Initialize parameters
    p = torch.randn(10240, device='cuda')
    grad = torch.randn(10240, device='cuda')
    exp_avg = torch.zeros(10240, device='cuda')

    # Define hyperparameters
    lr = 0.01
    wd = 0.01
    beta1 = 0.9
    beta2 = 0.999

    # Update parameters
    update_fn(p, grad, exp_avg, lr, wd, beta1, beta2)
