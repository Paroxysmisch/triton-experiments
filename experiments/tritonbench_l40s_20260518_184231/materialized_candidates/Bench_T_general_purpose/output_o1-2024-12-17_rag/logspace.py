import torch
import triton
import triton.language as tl

@triton.jit
def _logspace_kernel(
    out_ptr,      # Pointer to the output tensor
    start_val,    # Start value (float)
    end_val,      # End value (float)
    steps_val,    # Number of steps (int)
    base_val,     # Base of the logarithm (float)
    n_elements,   # Total number of elements
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # fraction = start + (end - start) * (offset / (steps - 1))
    fraction = start_val + (end_val - start_val) * (offsets / (steps_val - 1.0))
    values = tl.libdevice.powf(base_val, fraction)
    tl.store(out_ptr + offsets, values, mask=mask)

def logspace(start, end, steps, base=10.0, *, out=None, dtype=None, layout=torch.strided,
             device=None, requires_grad=False):
    if isinstance(start, torch.Tensor):
        start = start.item()
    if isinstance(end, torch.Tensor):
        end = end.item()

    if dtype is None:
        # Default to global default dtype if neither argument is complex
        dtype = torch.get_default_dtype()

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if out is not None:
        # Use the provided out tensor
        assert out.is_cuda, "Output tensor must be on CUDA for Triton."
        assert out.numel() == steps, "Output tensor size must match requested steps."
        out.requires_grad_(requires_grad)
    else:
        out = torch.empty((steps,), device=device, dtype=dtype, requires_grad=requires_grad)

    # Launch kernel
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(steps, meta["BLOCK_SIZE"]),)
    with torch.cuda.device(device):
        _logspace_kernel[grid](
            out, float(start), float(end), steps, float(base), steps,
            BLOCK_SIZE=BLOCK_SIZE
        )

    return out
