import torch
import triton
import triton.language as tl
from flag_gems.utils.shape_utils import volume

# Triton kernel to compute logarithmically spaced values
@triton.jit
def logspace_kernel(
    output_ptr,  # Pointer to the output tensor
    start,       # Starting value for the logarithmic scale
    end,         # Ending value for the logarithmic scale
    steps,       # Number of steps
    base,        # Base of the logarithm
    N,           # Total number of elements
    BLOCK_SIZE: tl.constexpr,  # Size of each block
):
    pid = tl.program_id(axis=0)  # Program ID for the current block
    block_start = pid * BLOCK_SIZE  # Start index for the current block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)  # Offsets for the current block
    mask = offsets < N  # Mask to ensure we don't write out of bounds

    # Calculate the logarithmically spaced values
    for i in range(BLOCK_SIZE):
        idx = offsets[i]
        if idx < N:
            value = base ** (start + (end - start) * idx / (steps - 1))
            tl.store(output_ptr + idx, value, mask=mask)

# Wrapper function for logspace
def logspace(start, end, steps, base=10.0, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False):
    if dtype is None:
        dtype = torch.get_default_dtype()  # Use default dtype if not specified
    if device is None:
        device = torch.device("cuda")  # Use CUDA device if not specified

    if isinstance(start, torch.Tensor):
        start = start.item()  # Convert to float if Tensor
    if isinstance(end, torch.Tensor):
        end = end.item()  # Convert to float if Tensor

    out = torch.empty(steps, device=device, dtype=dtype)  # Create an empty tensor
    N = steps  # Total number of elements
    grid_fn = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)  # Define grid size
    with torch.cuda.device(device):
        logspace_kernel[grid_fn](out, start, end, steps, base, N, BLOCK_SIZE=1024)  # Launch the kernel
    return out
