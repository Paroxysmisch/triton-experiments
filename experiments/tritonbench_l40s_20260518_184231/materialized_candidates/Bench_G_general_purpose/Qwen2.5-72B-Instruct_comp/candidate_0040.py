import triton
import triton.language as tl

@triton.jit
def masked_add_kernel(
    grad_ptr,  # Pointer to the gradient tensor
    p_ptr,     # Pointer to the tensor to be added
    p_mask_ptr,  # Pointer to the mask tensor
    n_elements,  # Total number of elements
    alpha,       # Scalar to multiply p_ptr
    BLOCK_SIZE: tl.constexpr  # Block size
):
    pid = tl.program_id(axis=0)  # Get the program ID
    block_start = pid * BLOCK_SIZE  # Calculate the starting index for this block

    # Create a mask to check bounds
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load elements from p_mask_ptr and convert to boolean
    p_mask = tl.load(p_mask_ptr + offsets, mask=mask).to(tl.int1)

    # Load elements from grad_ptr and p_ptr
    grad = tl.load(grad_ptr + offsets, mask=mask)
    p = tl.load(p_ptr + offsets, mask=mask)

    # Perform the masked addition
    result = tl.where(p_mask, grad + p * alpha, grad)

    # Store the result back in grad_ptr
    tl.store(grad_ptr + offsets, result, mask=mask)

import torch

def masked_add(grad, p, p_mask, alpha):
    # Assert that all tensors are on CUDA
    assert grad.is_cuda, "grad tensor must be on CUDA"
    assert p.is_cuda, "p tensor must be on CUDA"
    assert p_mask.is_cuda, "p_mask tensor must be on CUDA"

    # Assert that all tensors have the same layout and strides
    assert grad.stride() == p.stride(), "grad and p must have the same strides"
    assert grad.stride() == p_mask.stride(), "grad and p_mask must have the same strides"
    assert grad.shape == p.shape, "grad and p must have the same shape"
    assert grad.shape == p_mask.shape, "grad and p_mask must have the same shape"

    # Determine the grid size
    n_elements = grad.numel()
    BLOCK_SIZE = 256
    grid_size = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch the kernel
    masked_add_kernel[grid_size, BLOCK_SIZE](
        grad_ptr=grad,
        p_ptr=p,
        p_mask_ptr=p_mask,
        n_elements=n_elements,
        alpha=alpha
    )
