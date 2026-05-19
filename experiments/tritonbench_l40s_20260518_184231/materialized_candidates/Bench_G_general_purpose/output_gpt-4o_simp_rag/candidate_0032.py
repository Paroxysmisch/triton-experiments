import torch
import triton
import triton.language as tl

@triton.jit
def masked_add_kernel(grad_ptr,
                      p_ptr,
                      p_mask_ptr,
                      n_elements,
                      alpha,
                      BLOCK_SIZE: tl.constexpr):
    # Triton kernel to perform masked addition
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    # Load mask values and convert to boolean
    p_mask = tl.load(p_mask_ptr + offsets, mask=mask).to(tl.int1)
    # Invert mask to select elements where mask is 0
    mask = mask & ~p_mask
    # Load p and grad values with masking
    p = tl.load(p_ptr + offsets, mask=mask)
    grad = tl.load(grad_ptr + offsets, mask=mask)
    # Perform the masked addition
    grad += p * alpha
    # Store the result back to grad
    tl.store(grad_ptr + offsets, grad, mask=mask)

def masked_add_(grad: torch.Tensor, p_data: torch.Tensor, p_mask: torch.Tensor, alpha: float = 0):
    '''
    Function to call the Triton kernel for masked addition
    equivalent to
    grad.add_(p.data * (1 - p.mask), alpha=decay)
    '''
    # Ensure tensors are on CUDA
    assert grad.is_cuda and p_data.is_cuda and p_mask.is_cuda
    # Ensure tensors have the same layout and stride
    assert (grad.layout, p_data.layout, p_mask.layout) == (torch.strided, torch.strided, torch.strided)
    assert grad.stride() == p_data.stride() == p_mask.stride()
    # Number of elements to process
    n_elements = grad.numel()
    # Define the grid size for kernel execution
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    # Launch the kernel
    masked_add_kernel[grid](grad, p_data, p_mask, n_elements, alpha, BLOCK_SIZE=1024)

# Example usage
grad = torch.rand(1024, device='cuda')
p_data = torch.rand(1024, device='cuda')
p_mask = torch.randint(0, 2, (1024,), device='cuda', dtype=torch.int32)
alpha = 0.1

masked_add_(grad, p_data, p_mask, alpha)
