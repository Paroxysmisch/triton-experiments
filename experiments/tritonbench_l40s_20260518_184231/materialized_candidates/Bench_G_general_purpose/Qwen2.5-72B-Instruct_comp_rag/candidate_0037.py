import torch
import triton
import triton.language as tl

# Triton kernel to perform masked addition
@triton.jit
def masked_add_kernel(grad_ptr,
                      p_ptr,
                      p_mask_ptr,
                      n_elements,
                      alpha,
                      BLOCK_SIZE: tl.constexpr,
                      ):
    # Calculate the program ID
    pid = tl.program_id(axis=0)
    # Calculate the starting index for the block
    block_start = pid * BLOCK_SIZE
    # Generate offsets for the block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask to ensure we don't go out of bounds
    mask = offsets < n_elements
    # Load the mask values and convert to boolean
    p_mask = tl.load(p_mask_ptr + offsets, mask=mask).to(tl.int1)
    # Create a mask for elements to process
    mask = mask & ~p_mask
    # Load the data and gradient values
    p = tl.load(p_ptr + offsets, mask=mask)
    grad = tl.load(grad_ptr + offsets, mask=mask)
    # Perform the masked addition
    grad += p * alpha
    # Store the result back to the gradient tensor
    tl.store(grad_ptr + offsets, grad, mask=mask)

# Wrapper function to call the Triton kernel
def masked_add(grad: torch.Tensor, p_data: torch.Tensor, p_mask: torch.Tensor, alpha: float = 0):
    '''
    Function to call the Triton kernel for masked addition
    equivalent to
    grad.add_(p.data * (1 - p.mask), alpha=alpha)
    '''
    # Ensure all tensors are on CUDA
    assert grad.is_cuda and p_data.is_cuda and p_mask.is_cuda
    # Ensure all tensors have the same layout and strides
    assert (grad.layout, p_data.layout, p_mask.layout) == (torch.strided, torch.strided, torch.strided)
    assert grad.stride() == p_data.stride() == p_mask.stride()
    # Calculate the number of elements to process
    n_elements = grad.numel()
    # Calculate the grid size
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    # Launch the kernel
    masked_add_kernel[grid](grad, p_data, p_mask, n_elements, alpha, BLOCK_SIZE=1024)
