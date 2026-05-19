import torch
import triton
import triton.language as tl
import contextlib

@triton.jit
def sample_from_prob_kernel(
    x_ptr,         # *Pointer* to first input vector.
    output_ptr,    # *Pointer* to output vector.
    n_elements,    # Size of the vector.
    BLOCK_SIZE: tl.constexpr  # Number of elements each program should process.
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    output = x * 2

    tl.store(output_ptr + offsets, output, mask=mask)

def sample_from_prob(x: torch.Tensor):
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    sample_from_prob_kernel[grid](x, output, n_elements, BLOCK_SIZE=1024)
    return output

@contextlib.contextmanager
def autocast(device_type="cuda", enabled=True, dtype=None, cache_enabled=True):
    """
    Deprecated in favor of torch.amp.autocast("cuda"). This function allows scripts
    to run in mixed precision for forward passes and loss computation, choosing
    op-specific data types automatically. Avoid using it for backward passes.
    State is thread-local and can be nested with autocast(enabled=False) for
    forced precision in subregions. Each new thread must invoke this context
    manager or decorator on its own.
    """
    with torch.amp.autocast(device_type, enabled=enabled, dtype=dtype, cache_enabled=cache_enabled):
        yield
