import torch
import triton
import triton.language as tl

@triton.jit
def _logspace_kernel(OutPtr, start, end, base, steps, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Guard to avoid out-of-bounds
    mask = idx < steps
    
    # Handle single-step separately to avoid division by zero
    denominator = tl.where(steps > 1, steps - 1, 1)
    fraction = tl.where(steps > 1, (idx * (end - start) / denominator), 0.0)
    exponent = start + fraction
    out_val = tl.exp(tl.log(base) * exponent)
    
    # Write results
    tl.store(OutPtr + idx, out_val, mask=mask)

def logspace(start, end, steps, base=10.0, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False):
    # If start or end is a 0-dim tensor, extract its float value
    if isinstance(start, torch.Tensor):
        start = start.item()
    if isinstance(end, torch.Tensor):
        end = end.item()

    # Determine output dtype
    if dtype is None:
        # If neither start nor end is complex, fallback to default float dtype
        # (or matching complex float if needed)
        # Simplified to just handle float dtypes here
        dtype = torch.get_default_dtype()

    # Determine device
    if device is None:
        device = torch.device('cpu') if not torch.cuda.is_available() else torch.device('cuda')

    # Allocate out if None
    if out is None:
        out = torch.empty(steps, dtype=dtype, device=device, layout=layout, requires_grad=requires_grad)

    # If steps == 0 or steps < 0, let PyTorch handle the error
    if steps <= 0:
        raise ValueError("Number of steps must be > 0.")

    # If CPU, just do it with PyTorch to match typical fallback
    if device.type == 'cpu':
        if steps == 1:
            out[0] = base ** start
        else:
            step_size = (end - start) / (steps - 1)
            for i in range(steps):
                out[i] = base ** (start + i * step_size)
        return out

    # From here on, assume device.type == 'cuda'
    BLOCK_SIZE = 1024
    grid = ((steps + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    _logspace_kernel[grid](
        out, 
        start, 
        end, 
        base, 
        steps,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out
