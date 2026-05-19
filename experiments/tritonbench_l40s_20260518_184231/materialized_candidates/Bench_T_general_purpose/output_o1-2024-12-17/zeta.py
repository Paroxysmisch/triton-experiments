import torch
import triton
import triton.language as tl

# Kernel to compute an approximation of the Hurwitz zeta function elementwise:
# zeta(x, q) = sum_{k=0}^{∞} 1 / (k + q)^x
# For demonstration purposes, we use a fixed number of iterations (MAX_ITER).

@triton.jit
def _zeta_kernel(
    x_ptr, q_ptr, out_ptr,
    n_elements, MAX_ITER,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x_vals = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    q_vals = tl.load(q_ptr + offsets, mask=mask, other=0.0)

    acc = tl.zeros_like(x_vals)
    # Naive sum of partial series
    for k in range(MAX_ITER):
        num = 1.0
        denom = (q_vals + k) ** x_vals
        acc += num / denom

    tl.store(out_ptr + offsets, acc, mask=mask)

def zeta(input: torch.Tensor, other: torch.Tensor, *, out: torch.Tensor = None) -> torch.Tensor:
    # Ensure input and other have the same shape for elementwise operation
    if input.shape != other.shape:
        raise ValueError("Input shapes must match.")
    
    # Allocate output if not provided
    if out is None:
        out = torch.empty_like(input)
    
    # Flatten tensors for Triton kernel launch
    x_data = input.contiguous().flatten()
    q_data = other.contiguous().flatten()
    out_data = out.contiguous().flatten()

    n_elements = x_data.numel()
    BLOCK_SIZE = 1024
    # Choose how many terms in the partial sum (for demonstration)
    MAX_ITER = 100

    # Launch kernel
    grid = lambda meta: ( (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE, )
    _zeta_kernel[grid](
        x_data, q_data, out_data,
        n_elements, MAX_ITER,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return out.reshape(input.shape)
