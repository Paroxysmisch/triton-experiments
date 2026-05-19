import triton
import triton.language as tl

# Define the block size
BLOCK_SIZE = 256

# Triton kernel definition
@triton.jit
def masked_add_kernel(
    grad_ptr, p_ptr, p_mask_ptr, n_elements, alpha,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)  # unique ID for each block
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements  # mask to ensure we don't access out of bounds

    # Load data
    grad = tl.load(grad_ptr + offsets, mask=mask)
    p = tl.load(p_ptr + offsets, mask=mask)
    p_mask = tl.load(p_mask_ptr + offsets, mask=mask)

    # Convert p_mask to boolean mask
    p_mask_bool = tl.bitcast(p_mask, tl.int1)

    # Perform element-wise addition
    result = grad + alpha * p * p_mask_bool

    # Store the result back
    tl.store(grad_ptr + offsets, result, mask=mask)

# Wrapper function to invoke the kernel
def masked_add(grad, p, p_mask, alpha):
    # Assertions to ensure tensors are on CUDA, with the same layout and strides
    assert grad.device.type == 'cuda'
    assert p.device.type == 'cuda'
    assert p_mask.device.type == 'cuda'
    assert grad.layout == p.layout == p_mask.layout
    assert grad.stride(0) == p.stride(0) == p_mask.stride(0)

    # Determine the grid size
    n_elements = grad.shape[0]
    grid_size = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch the kernel
    masked_add_kernel[grid_size, BLOCK_SIZE](grad.data_ptr(), p.data_ptr(), p_mask.data_ptr(), n_elements, alpha)

# Example usage
if __name__ == "__main__":
    import torch

    # Example tensors
    grad = torch.randn(1024, device='cuda')
    p = torch.randn(1024, device='cuda')
    p_mask = torch.randint(0, 2, (1024,), device='cuda')
    alpha = 0.5

    # Call the wrapper function
    masked_add(grad, p, p_mask, alpha)

    # Print the result
    print(grad)
