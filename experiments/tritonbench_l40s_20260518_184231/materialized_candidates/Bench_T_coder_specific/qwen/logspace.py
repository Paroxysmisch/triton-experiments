import triton
import torch

@triton.jit
def logspace_kernel(output_ptr, start, end, steps, base, BLOCK_SIZE: int = 256):
    pid = tl.program_id(axis=0)
    coords = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = coords < steps
    start = tl.load(start)
    end = tl.load(end)
    steps = tl.load(steps)
    base = tl.load(base)

    linear_index = coords
    step_size = (end - start) / (steps - 1)
    value = tl.pow(base, start + linear_index * step_size)
    tl.store(output_ptr + coords, value, mask=mask)

def logspace(start, end, steps, base=10.0, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False):
    # Validate inputs
    if not isinstance(start, (float, torch.Tensor)) or not isinstance(end, (float, torch.Tensor)):
        raise ValueError("start and end must be float or Tensor")
    if not isinstance(steps, int):
        raise ValueError("steps must be an integer")
    if not isinstance(base, (float, torch.Tensor)):
        raise ValueError("base must be float or Tensor")

    # Convert inputs to Triton tensors
    if isinstance(start, float):
        start = torch.tensor([start], dtype=torch.float32, device=device)
    if isinstance(end, float):
        end = torch.tensor([end], dtype=torch.float32, device=device)
    if isinstance(base, float):
        base = torch.tensor([base], dtype=torch.float32, device=device)

    # Create output tensor if not provided
    if out is None:
        out = torch.empty((steps,), dtype=dtype or torch.float32, layout=layout, device=device, requires_grad=requires_grad)

    # Launch Triton kernel
    block_size = 256
    grid_size = (out.numel() + block_size - 1) // block_size
    logspace_kernel[grid_size, block_size](out.data_ptr(), start.item(), end.item(), steps, base.item())

    return out
