import torch
import triton
import triton.language as tl


@triton.jit
def _zeta_kernel(
    x_ptr,    # Pointer to tensor representing 'x'
    q_ptr,    # Pointer to tensor representing 'q'
    out_ptr,  # Pointer to output tensor
    n_elements,  # Total number of elements
    n_terms,     # Number of terms to approximate the infinite series
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load x and q for each element
    x_vals = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    q_vals = tl.load(q_ptr + offsets, mask=mask, other=0.0)

    # Compute the partial sum of the Hurwitz zeta function
    zeta_sum = 0.0
    # Rudimentary series approximation up to n_terms
    for k in range(n_terms):
        term = 1.0 / tl.pow(q_vals + k, x_vals)
        zeta_sum += term

    # Store results
    tl.store(out_ptr + offsets, zeta_sum, mask=mask)


def zeta(input: torch.Tensor, other: torch.Tensor, *, out: torch.Tensor = None) -> torch.Tensor:
    """
    zeta(input, other, *, out=None) -> Tensor
    Approximates the Hurwitz zeta function elementwise:
      ζ(x, q) = sum_{k=0}^{∞} [ 1 / (k + q)^x ]
    Riemann zeta is the special case q=1.
    """
    # Broadcast if needed
    x_b, q_b = torch.broadcast_tensors(input, other)

    # Ensure CUDA device
    assert x_b.is_cuda and q_b.is_cuda, "Inputs must be on CUDA device."

    # Allocate out if None
    if out is None:
        out = torch.empty_like(x_b)

    n_elements = x_b.numel()
    # Approximation terms for the series
    N_TERMS = 100

    # Choose a BLOCK_SIZE (power of 2 for efficiency)
    BLOCK_SIZE = 256
    grid = ( (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE, )

    _zeta_kernel[grid](
        x_ptr=x_b,
        q_ptr=q_b,
        out_ptr=out,
        n_elements=n_elements,
        n_terms=N_TERMS,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return out
