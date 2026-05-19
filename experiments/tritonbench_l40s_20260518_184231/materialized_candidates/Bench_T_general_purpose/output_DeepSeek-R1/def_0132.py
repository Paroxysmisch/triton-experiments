import torch
import triton
import triton.language as tl

@triton.jit
def mul_sub_kernel(
    input_ptr, other_mul_ptr, other_sub_ptr, out_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    input = tl.load(input_ptr + offsets, mask=mask)
    other_mul = tl.load(other_mul_ptr + offsets, mask=mask)
    other_sub = tl.load(other_sub_ptr + offsets, mask=mask)

    output = input * other_mul - other_sub

    tl.store(out_ptr + offsets, output, mask=mask)

def mul_sub(input, other_mul, other_sub, alpha=1, out=None) -> torch.Tensor:
    # Convert scalars to tensors and ensure they are on the same device and dtype as input
    if not isinstance(other_mul, torch.Tensor):
        other_mul = torch.tensor(other_mul, device=input.device, dtype=input.dtype)
    if not isinstance(other_sub, torch.Tensor):
        other_sub = torch.tensor(other_sub, device=input.device, dtype=input.dtype)
    
    # Scale other_sub by alpha
    other_sub_scaled = other_sub * alpha

    # Compute the broadcasted shape
    try:
        # First, compute the shape after input * other_mul
        shape_mul = torch.broadcast_shapes(input.shape, other_mul.shape)
        # Then, compute the final shape after subtracting other_sub_scaled
        final_shape = torch.broadcast_shapes(shape_mul, other_sub_scaled.shape)
    except RuntimeError as e:
        raise RuntimeError("Inputs are not broadcastable") from e

    # Expand all tensors to the final broadcasted shape
    input_expanded = input.expand(final_shape)
    other_mul_expanded = other_mul.expand(final_shape)
    other_sub_scaled_expanded = other_sub_scaled.expand(final_shape)

    # Ensure output tensor is properly allocated
    if out is None:
        out = torch.empty(final_shape, device=input.device, dtype=input.dtype)
    else:
        if out.shape != final_shape:
            raise RuntimeError("Output tensor shape does not match the broadcasted shape")
        if not out.is_contiguous():
            raise RuntimeError("Output tensor must be contiguous")

    # Flatten the tensors for 1D kernel processing
    input_flat = input_expanded.contiguous().view(-1)
    other_mul_flat = other_mul_expanded.contiguous().view(-1)
    other_sub_scaled_flat = other_sub_scaled_expanded.contiguous().view(-1)
    out_flat = out.contiguous().view(-1)

    n_elements = out_flat.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    mul_sub_kernel[grid](
        input_flat, other_mul_flat, other_sub_scaled_flat, out_flat,
        n_elements, BLOCK_SIZE=1024
    )

    return out
