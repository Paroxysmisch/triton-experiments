import triton
import triton.language as tl
import torch

@triton.jit
def _add_kernel(A, B, C, size, BLOCK: tl.constexpr):
    # Get the current program index
    prog_id = tl.program_id(0)
    # Calculate offsets for this program
    offs = prog_id * BLOCK + tl.arange(0, BLOCK)
    # Load data from A and B with a mask
    a_vals = tl.load(A + offs, mask=offs < size)
    b_vals = tl.load(B + offs, mask=offs < size)
    # Compute the addition
    c_vals = a_vals + b_vals
    # Store the result in C
    tl.store(C + offs, c_vals, mask=offs < size)

def custom_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # Initialize c as an empty tensor
    c = torch.empty_like(a)
    # Compute size
    size = a.numel()
    # Set BLOCK size
    BLOCK = 16
    # Calculate grid size for kernel launch
    grid = lambda meta: (triton.cdiv(size, meta['BLOCK']), )
    # Launch the Triton kernel with the calculated grid
    _add_kernel[grid](a, b, c, size, BLOCK=BLOCK)
    # Return the result
    return c
