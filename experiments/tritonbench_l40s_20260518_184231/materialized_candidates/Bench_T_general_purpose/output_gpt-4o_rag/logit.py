import triton
import triton.language as tl
import torch

@triton.jit
def logit_kernel(input_ptr, output_ptr, n_elements, eps_ptr, BLOCK_SIZE: tl.constexpr):
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < n_elements
    
    # Load input
    x = tl.load(input_ptr + idx, mask=mask)
    
    # Load epsilon if not None
    eps = tl.load(eps_ptr) if eps_ptr else None
    
    # Clamp input if epsilon is provided
    if eps:
        x = tl.where(x < eps, eps, x)
        x = tl.where(x > 1 - eps, 1 - eps, x)
    
    # Compute logit
    z = tl.where((x >= 0) & (x <= 1), x, float('nan'))
    y = tl.log(z / (1 - z))
    
    # Store result
    tl.store(output_ptr + idx, y, mask=mask)

def logit(input, eps=None, *, out=None):
    # Prepare input tensor
    input_flat = input.flatten()
    n_elements = input_flat.numel()
    
    # Prepare output tensor
    if out is None:
        out = torch.empty_like(input_flat)
    else:
        out = out.flatten()
    
    # Prepare epsilon tensor
    eps_tensor = torch.tensor(eps, device=input.device) if eps is not None else None
    
    # Define block size
    BLOCK_SIZE = 1024  # Adjust according to your hardware capabilities
    
    # Launch kernel
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    logit_kernel[grid](
        input_flat,
        out,
        n_elements,
        eps_tensor,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Reshape output to match input
    return out.view_as(input)

# Example usage
input_tensor = torch.tensor([0.1, 0.5, 0.9, 1.1, -0.1], device='cuda')
output_tensor = logit(input_tensor, eps=0.01)
print(output_tensor)
