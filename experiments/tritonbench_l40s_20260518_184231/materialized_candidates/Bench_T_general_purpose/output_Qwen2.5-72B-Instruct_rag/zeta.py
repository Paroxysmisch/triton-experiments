import torch
import triton
import triton.language as tl

device = 'cuda:0'

@triton.jit
def zeta_kernel(
    x_ptr,
    q_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    q = tl.load(q_ptr + offsets, mask=mask)

    # Initialize the result with zeros
    result = tl.zeros_like(x)

    # Compute the Hurwitz zeta function for each element
    k = 0
    while True:
        term = tl.where(k + q > 0, 1.0 / (k + q) ** x, 0.0)
        result += term
        k += 1
        if tl.all(term < 1e-10, axis=0):
            break

    tl.store(out_ptr + offsets, result, mask=mask)

def zeta(input: torch.Tensor, other: torch.Tensor, *, out: torch.Tensor = None) -> torch.Tensor:
    assert input.is_cuda and other.is_cuda, 'Input tensors must be on GPU'
    assert input.shape == other.shape, 'Input tensors must have the same shape'

    if out is None:
        out = torch.empty_like(input).to(device)

    n_elements = input.numel()
    BLOCK_SIZE = triton.next_power_of_2(n_elements)

    zeta_kernel[(n_elements // BLOCK_SIZE + 1,)](
        x_ptr=input, q_ptr=other, out_ptr=out, n_elements=n_elements, BLOCK_SIZE=BLOCK_SIZE
    )

    return out
