import torch
import triton
import triton.language as tl

# Triton kernel to perform masked addition
@triton.jit
def masked_add_kernel(
    grad_ptr,
    p_ptr,
    p_mask_ptr,
    n_elements,
    alpha,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the program ID
    pid = tl.program_id(axis=0)
    # Calculate the start index for the block
    block_start = pid * BLOCK_SIZE
    # Generate the offsets for the block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask to ensure we don't access out-of-bounds elements
    mask = offsets < n_elements
    # Load the mask values
    p_mask = tl.load(p_mask_ptr + offsets, mask=mask).to(tl.int1)
    # Apply the mask to skip elements where p_mask is 1
    mask = mask & ~p_mask
    # Load the data and gradient values
    p = tl.load(p_ptr + offsets, mask=mask)
    grad = tl.load(grad_ptr + offsets, mask=mask)
    # Perform the masked addition
    grad += p * alpha
    # Store the result back to the gradient tensor
    tl.store(grad_ptr + offsets, grad, mask=mask)

# Wrapper function to call the Triton kernel
def masked_add_(grad: torch.Tensor, p_data: torch.Tensor, p_mask: torch.Tensor, alpha: float = 0):
    '''
    Function to call the Triton kernel for masked addition
    equivalent to
    grad.add_(p.data * (1 - p.mask), alpha=alpha)
    '''
    # Ensure all tensors are on CUDA
    assert grad.is_cuda and p_data.is_cuda and p_mask.is_cuda
    # Ensure all tensors have the same layout and stride
    assert (grad.layout, p_data.layout, p_mask.layout) == (torch.strided, torch.strided, torch.strided)
    assert grad.stride() == p_data.stride() == p_mask.stride()
    # Calculate the number of elements
    n_elements = grad.numel()
    # Define the grid size for the kernel
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    # Call the Triton kernel
    masked_add_kernel[grid](grad, p_data, p_mask, n_elements, alpha, BLOCK_SIZE=1024)
