import torch
import triton
import triton.language as tl

# Triton kernel for ELU activation
@triton.jit
def elu_kernel(
    x_ptr,          # *Pointer* to input vector
    alpha_ptr,      # *Pointer* to alpha scalar
    output_ptr,     # *Pointer* to output vector
    n_elements,     # Total number of elements
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    alpha = tl.load(alpha_ptr)
    out = tl.where(x > 0, x, alpha * (tl.exp(x) - 1))
    tl.store(output_ptr + offsets, out, mask=mask)


def elu(x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    elu_kernel[grid](x, alpha, output, n_elements, BLOCK_SIZE=1024)
    return output


def elu_inplace(x: torch.Tensor, alpha: torch.Tensor):
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    elu_kernel[grid](x, alpha, x, n_elements, BLOCK_SIZE=1024)


def elu_linear(input: torch.Tensor,
               weight: torch.Tensor,
               bias: torch.Tensor = None,
               alpha: float = 1.0,
               inplace: bool = False) -> torch.Tensor:
    # Linear transformation
    out = input.matmul(weight)
    if bias is not None:
        out = out + bias

    # Prepare alpha as a single-element tensor on the same device
    alpha_t = torch.tensor([alpha], dtype=out.dtype, device=out.device)

    # ELU activation
    if inplace:
        elu_inplace(out, alpha_t)
        return out
    else:
        return elu(out, alpha_t)
