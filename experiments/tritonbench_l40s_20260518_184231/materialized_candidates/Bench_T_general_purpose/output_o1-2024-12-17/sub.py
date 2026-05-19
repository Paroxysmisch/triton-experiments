import triton
import triton.language as tl

@triton.jit
def _sub_kernel(
    input_ptr, 
    other_ptr,
    out_ptr,
    alpha,
    n_elements, 
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    in_val = tl.load(input_ptr + offsets, mask=mask, other=0)
    other_val = tl.load(other_ptr + offsets, mask=mask, other=0)
    result = in_val - alpha * other_val
    tl.store(out_ptr + offsets, result, mask=mask)

def sub(input, other, *, alpha=1, out=None):
    # Ensure input and other are Triton-compatible tensors
    # (Assuming input and other are already on device or
    #  made compatible through other means)
    import torch

    # Convert scalars to Tensors if necessary
    if not isinstance(input, torch.Tensor):
        input = torch.tensor(input)
    if not isinstance(other, torch.Tensor):
        other = torch.tensor(other)

    # Broadcast to a common shape
    common_shape = torch.broadcast_shapes(input.shape, other.shape)
    input_b = input.expand(common_shape)
    other_b = other.expand(common_shape)

    # Type promotion
    dtype = torch.promote_types(input_b.dtype, other_b.dtype)
    input_b = input_b.to(dtype)
    other_b = other_b.to(dtype)

    # Prepare output
    if out is None:
        out = torch.empty(common_shape, dtype=dtype, device=input_b.device)

    # Flatten for kernel launch
    input_flat = input_b.contiguous().view(-1)
    other_flat = other_b.contiguous().view(-1)
    out_flat = out.contiguous().view(-1)
    n_elements = out_flat.numel()

    # Define block size
    BLOCK_SIZE = 1024
    grid = lambda meta: ( (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE, )

    _sub_kernel[grid](
        input_flat,
        other_flat,
        out_flat,
        alpha,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out
