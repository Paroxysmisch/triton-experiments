import torch
import triton
import triton.language as tl

@triton.jit
def scaled_add_kernel(
    y_ptr,  # Pointer to the y tensor
    x_ptr,  # Pointer to the x tensor
    alpha_ptr,  # Pointer to the alpha scalar tensor
    n_elements,  # Number of elements in the vectors
    BLOCK_SIZE: tl.constexpr,  # Number of elements each block should process
):
    # Load the alpha value once from the scalar tensor
    alpha = tl.load(alpha_ptr)
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load the current elements of x and y
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # Compute the scaled addition
    y_updated = y + alpha * x

    # Store the updated y values back to memory
    tl.store(y_ptr + offsets, y_updated, mask=mask)

def scaled_add_norm(y: torch.Tensor, x: torch.Tensor, alpha: float) -> torch.Tensor:
    # Ensure y and x are 1D tensors of the same shape and on CUDA
    assert y.dim() == 1, "y must be a 1D tensor"
    assert x.dim() == 1, "x must be a 1D tensor"
    assert y.shape == x.shape, "y and x must have the same shape"
    assert y.is_cuda and x.is_cuda, "Tensors must be on CUDA device"
    assert y.is_contiguous(), "y must be contiguous"
    assert x.is_contiguous(), "x must be contiguous"

    n_elements = y.numel()
    # Convert alpha to a tensor with the same dtype and device as y
    alpha_tensor = torch.tensor(alpha, dtype=y.dtype, device=y.device)

    # Configure the kernel grid and launch parameters
    BLOCK_SIZE = 1024  # Adjust based on optimal performance for the hardware
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # Launch the kernel to perform y += alpha * x
    scaled_add_kernel[grid](y, x, alpha_tensor, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    # Ensure all CUDA operations are completed before computing the norm
    torch.cuda.synchronize(y.device)

    # Calculate and return the 2-norm of the updated y
    return torch.norm(y)
