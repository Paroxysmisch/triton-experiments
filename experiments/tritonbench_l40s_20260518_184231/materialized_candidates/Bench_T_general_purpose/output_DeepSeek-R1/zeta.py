import torch
import triton
import triton.language as tl
from typing import Optional

@triton.jit
def zeta_kernel(
    input_ptr,
    other_ptr,
    output_ptr,
    input_stride,
    other_stride,
    output_stride,
    num_elements,
    K: tl.constexpr,  # Number of terms to sum, compile-time constant
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements

    # Load x and q for the current block
    x = tl.load(input_ptr + offsets * input_stride, mask=mask)
    q = tl.load(other_ptr + offsets * other_stride, mask=mask)

    total = tl.zeros_like(x)
    for k in range(K):
        term = 1.0 / tl.pow((k + q), x)
        total += term

    tl.store(output_ptr + offsets * output_stride, total, mask=mask)

def zeta(input: torch.Tensor, other: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    # Broadcast input and other to the same shape
    try:
        torch.broadcast_shapes(input.shape, other.shape)
    except ValueError as e:
        raise ValueError(f"input and other are not broadcastable: {e}") from e
    
    broadcasted_input, broadcasted_other = torch.broadcast_tensors(input, other)
    input_contig = broadcasted_input.contiguous()
    other_contig = broadcasted_other.contiguous()
    output_shape = input_contig.shape

    # Prepare output tensor
    if out is None:
        output = torch.empty_like(input_contig)
    else:
        if out.shape != output_shape:
            raise ValueError("out tensor shape does not match broadcasted input shape")
        output = out
    output_contig = output.contiguous()

    # Flatten tensors to 1D for kernel processing
    input_flat = input_contig.view(-1)
    other_flat = other_contig.view(-1)
    output_flat = output_contig.view(-1)
    num_elements = input_flat.numel()

    if num_elements == 0:
        return output

    # Kernel configuration
    K = 1000  # Fixed number of terms for the approximation
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(num_elements, meta['BLOCK_SIZE']),)
    
    zeta_kernel[grid](
        input_flat.data_ptr(),
        other_flat.data_ptr(),
        output_flat.data_ptr(),
        input_flat.stride(0),
        other_flat.stride(0),
        output_flat.stride(0),
        num_elements,
        K=K,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output
