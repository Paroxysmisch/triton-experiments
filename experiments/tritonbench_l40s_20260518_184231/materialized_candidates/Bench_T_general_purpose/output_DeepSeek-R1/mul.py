import torch
import triton
import triton.language as tl

@triton.jit
def mul_kernel(
    input_ptr, other_ptr, output_ptr,
    n_elements,
    other_numel: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    input = tl.load(input_ptr + offsets, mask=mask)
    if other_numel == 1:
        other = tl.load(other_ptr)
        output = input * other
    else:
        other = tl.load(other_ptr + offsets, mask=mask)
        output = input * other
    tl.store(output_ptr + offsets, output, mask=mask)

def mul(input, other, *, out=None):
    # Handle scalar 'other' and type promotion
    if not isinstance(other, torch.Tensor):
        other = torch.tensor(other, device=input.device)
    # Ensure both are on the same device
    assert input.device == other.device, "Input and other must be on the same device"
    # Compute promoted dtype
    promoted_dtype = torch.promote_types(input.dtype, other.dtype)
    input = input.to(dtype=promoted_dtype)
    other = other.to(dtype=promoted_dtype)
    # Broadcast tensors
    try:
        input_bc, other_bc = torch.broadcast_tensors(input, other)
    except RuntimeError as e:
        raise RuntimeError(f"Shapes {input.shape} and {other.shape} are not broadcastable") from e
    # Flatten to 1D contiguous tensors
    input_flat = input_bc.contiguous().view(-1)
    other_flat = other_bc.contiguous().view(-1)
    # Prepare output
    if out is None:
        out = torch.empty_like(input_flat)
    else:
        assert out.is_contiguous(), "Output tensor must be contiguous"
        out = out.view(-1)
    other_numel = other_flat.numel()
    n_elements = input_flat.numel()
    # Kernel grid setup
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    mul_kernel[grid](input_flat, other_flat, out, n_elements, other_numel, BLOCK_SIZE=1024)
    # Reshape output to broadcasted shape
    return out.view_as(input_bc)
