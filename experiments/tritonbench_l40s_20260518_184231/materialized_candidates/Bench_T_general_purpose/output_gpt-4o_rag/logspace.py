import torch
import triton
import triton.language as tl

# Triton kernel to generate logarithmically spaced values
@triton.jit
def logspace_kernel(
    output_ptr,  # Pointer to the output tensor
    start,       # Starting value for the set of points
    end,         # Ending value for the set of points
    base,        # Base of the logarithm function
    steps,       # Number of steps
    BLOCK_SIZE: tl.constexpr  # Size of each block
):
    pid = tl.program_id(axis=0)  # Program ID for the current block
    block_start = pid * BLOCK_SIZE  # Start index for the current block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)  # Offsets for the current block
    mask = offsets < steps  # Mask to ensure we don't write out of bounds
    
    # Calculate the logarithmically spaced values
    delta = (end - start) / (steps - 1)
    exponents = start + offsets * delta
    values = base ** exponents
    tl.store(output_ptr + offsets, values, mask=mask)  # Store the calculated values

# Wrapper function to create a tensor with logarithmically spaced values
def logspace(start, end, steps, base=10.0, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False):
    if dtype is None:
        dtype = torch.get_default_dtype()  # Use default dtype if not specified
    if device is None:
        device = torch.device("cuda")  # Use CUDA device if not specified

    if isinstance(start, torch.Tensor):
        start = start.item()  # Extract scalar value if start is a tensor
    if isinstance(end, torch.Tensor):
        end = end.item()  # Extract scalar value if end is a tensor

    if out is None:
        out = torch.empty(steps, device=device, dtype=dtype)  # Create an empty tensor if out is not provided

    grid_fn = lambda meta: (triton.cdiv(steps, meta["BLOCK_SIZE"]),)  # Define grid size
    with torch.cuda.device(device):
        logspace_kernel[grid_fn](out, start, end, base, steps, BLOCK_SIZE=1024)  # Launch the kernel
    
    if requires_grad:
        out.requires_grad_()
    
    return out
