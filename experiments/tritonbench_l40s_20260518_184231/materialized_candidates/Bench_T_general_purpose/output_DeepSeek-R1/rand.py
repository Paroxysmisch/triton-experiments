import torch
import triton
import triton.language as tl

@triton.jit
def rand_kernel(output_ptr, n_elements, seed, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Generate random numbers using the seed and offsets
    # Note: tl.rand uses a seed based on the launch grid and offsets to ensure uniqueness
    random = tl.rand(seed, offsets)
    tl.store(output_ptr + offsets, random, mask=mask)

def rand(*size, generator=None, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False, pin_memory=False):
    # Process variable size arguments
    shape = torch.Size(size)
    
    # Determine device and dtype
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if dtype is None:
        dtype = torch.get_default_dtype()
    
    # Create output tensor if not provided
    if out is None:
        out = torch.empty(shape, dtype=dtype, layout=layout, device=device, pin_memory=pin_memory)
    else:
        if out.shape != shape:
            raise ValueError(f"out tensor has shape {out.shape} which does not match desired shape {shape}")
        # Ensure the out tensor is on the correct device and dtype
        if out.device != device:
            raise ValueError(f"out tensor device {out.device} does not match requested device {device}")
        if out.dtype != dtype:
            raise ValueError(f"out tensor dtype {out.dtype} does not match requested dtype {dtype}")
    
    # Handle generator to get seed
    if generator is not None:
        # Generate a random seed using the provided generator
        seed = torch.empty((), dtype=torch.int64, device='cpu').random_(generator=generator).item()
    else:
        # Generate a default seed (not recommended for production)
        seed = 1234
    
    # Launch kernel only if there are elements to fill
    n_elements = out.numel()
    if n_elements > 0:
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        rand_kernel[grid](out, n_elements, seed, BLOCK_SIZE=1024)
    
    # Set autograd options
    out.requires_grad_(requires_grad)
    return out
