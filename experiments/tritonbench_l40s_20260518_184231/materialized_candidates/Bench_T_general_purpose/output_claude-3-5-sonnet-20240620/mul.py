{{ code }}
import triton
import triton.language as tl

@triton.jit
def kernel_mul(input_ptr, other_ptr, out_ptr, N, stride_input, stride_other, stride_out):
    # Compute the index for each thread
    idx = tl.program_id(0) * tl.block_size(0) + tl.arange(0, tl.block_size(0))
    mask = idx < N  # Mask to avoid out-of-bounds access

    # Load input and other tensors
    input_val = tl.load(input_ptr + idx * stride_input, mask=mask)
    other_val = tl.load(other_ptr + idx * stride_other, mask=mask)

    # Perform multiplication
    out_val = input_val * other_val

    # Store the result
    tl.store(out_ptr + idx * stride_out, out_val, mask=mask)

def mul(input: torch.Tensor, other: Union[torch.Tensor, float, int], *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    # Determine the output tensor shape
    if out is None:
        out_shape = torch.broadcast_shapes(input.shape, torch.tensor(other).shape) if isinstance(other, torch.Tensor) else input.shape
        out = torch.empty(out_shape, dtype=torch.promote_types(input.dtype, torch.tensor(other).dtype if isinstance(other, torch.Tensor) else torch.tensor(other).dtype))

    # Get the number of elements
    N = out.numel()

    # Launch the Triton kernel
    grid = (triton.cdiv(N, 1024),)
    kernel_mul[grid](input, other, out, N, input.stride(0), other.stride(0) if isinstance(other, torch.Tensor) else 0, out.stride(0))

    return out
{{ code }}
