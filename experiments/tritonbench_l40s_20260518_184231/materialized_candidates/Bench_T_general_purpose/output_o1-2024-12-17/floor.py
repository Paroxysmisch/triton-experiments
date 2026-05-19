import triton
import triton.language as tl
import torch

@triton.jit
def _floor_kernel(in_ptr, out_ptr, n, BLOCKSIZE: tl.constexpr):
    pid = tl.program_id(0)
    start = pid * BLOCKSIZE
    offsets = start + tl.arange(0, BLOCKSIZE)
    mask = offsets < n
    x = tl.load(in_ptr + offsets, mask=mask)
    floored = tl.floor(x)
    tl.store(out_ptr + offsets, floored, mask=mask)

def floor(input, *, out=None):
    # If integer inputs, return a copy per array-api convention
    if not torch.is_floating_point(input):
        if out is not None:
            return out.copy_(input)
        return input.clone()

    if out is None:
        out = torch.empty_like(input)

    # Ensure contiguous tensors for simplicity
    in_contig = input.contiguous()
    out_contig = out.contiguous()

    assert in_contig.is_cuda, "Input tensor must be on CUDA device."
    assert out_contig.is_cuda, "Output tensor must be on CUDA device."

    n = in_contig.numel()
    BLOCKSIZE = 1024
    grid = lambda meta: ( (n + BLOCKSIZE - 1) // BLOCKSIZE, )

    _floor_kernel[grid](
        in_contig.data_ptr(),
        out_contig.data_ptr(),
        n,
        BLOCKSIZE=BLOCKSIZE
    )

    # If the original out was provided and not contiguous, copy back
    if out is not out_contig:
        out.copy_(out_contig)
    return out
