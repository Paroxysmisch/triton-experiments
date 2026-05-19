import torch
import triton
import triton.language as tl
import math

@triton.jit
def logspace_kernel(
    output_ptr,
    start_val,
    delta_val,
    ln_base,
    steps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < steps
    i = offsets
    exponent = start_val + i * delta_val
    value = tl.exp(exponent * ln_base)
    tl.store(output_ptr + offsets, value, mask=mask)

def logspace(start, end, steps, base=10.0, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False):
    # Process start and end values
    def get_value(x):
        if isinstance(x, torch.Tensor):
            assert x.ndim == 0, "start/end must be 0-dimensional tensors"
            return x.item()
        return x

    start_val = get_value(start)
    end_val = get_value(end)

    # Calculate delta
    if steps > 1:
        delta_val = (end_val - start_val) / (steps - 1)
    else:
        delta_val = 0.0

    # Determine dtype
    if dtype is None:
        is_complex = isinstance(start_val, complex) or isinstance(end_val, complex)
        if is_complex:
            dtype = torch.get_default_complex_dtype()
        else:
            dtype = torch.get_default_dtype()
    else:
        is_complex = dtype.is_complex

    # Handle complex dtypes with PyTorch's implementation
    if is_complex or (isinstance(start_val, complex) or isinstance(end_val, complex)):
        return torch.logspace(
            start_val.real if isinstance(start_val, complex) else start_val,
            end_val.real if isinstance(end_val, complex) else end_val,
            steps,
            base=base,
            dtype=dtype,
            layout=layout,
            device=device,
            requires_grad=requires_grad,
            out=out
        )

    # Create output tensor
    if out is not None:
        assert out.size(0) == steps, "out tensor must have size steps"
        assert out.is_floating_point(), "out tensor must be of floating point type"
        output = out
    else:
        output = torch.empty(steps, dtype=dtype, device=device, layout=layout)

    # Check if CUDA tensor, else use PyTorch's CPU implementation
    if not output.is_cuda:
        return torch.logspace(
            start_val,
            end_val,
            steps,
            base=base,
            dtype=dtype,
            layout=layout,
            device=device,
            requires_grad=requires_grad,
            out=out
        )

    # Cast parameters to output's dtype
    dtype = output.dtype
    start_val = torch.tensor(start_val, dtype=dtype).item()
    delta_val = torch.tensor(delta_val, dtype=dtype).item()
    ln_base_val = math.log(base)
    ln_base = torch.tensor(ln_base_val, dtype=dtype).item()

    # Launch Triton kernel
    N = steps
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    with torch.cuda.device(output.device):
        logspace_kernel[grid](output, start_val, delta_val, ln_base, N, BLOCK_SIZE=BLOCK_SIZE)

    output.requires_grad_(requires_grad)
    return output
