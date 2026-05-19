import torch
import triton
import triton.language as tl

@triton.jit
def add_mean_kernel(
    input_ptr, other_ptr, output_ptr, alpha, num_elements,
    input_stride, other_stride, output_stride,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements

    # Load input and other tensors
    input = tl.load(input_ptr + offsets * input_stride, mask=mask)
    other = tl.load(other_ptr + offsets * other_stride, mask=mask)

    # Perform the operation: input + alpha * other
    result = input + alpha * other

    # Store the result
    tl.store(output_ptr + offsets * output_stride, result, mask=mask)

def add_mean(input, other, dim=None, alpha=1, keepdim=False, dtype=None, out=None):
    # Ensure input and other are tensors
    if not isinstance(input, torch.Tensor):
        raise TypeError("input must be a torch.Tensor")
    if not isinstance(other, torch.Tensor):
        other = torch.tensor(other, dtype=input.dtype, device=input.device)

    # Type promotion
    if dtype is not None:
        input = input.to(dtype)
        other = other.to(dtype)

    # Broadcast input and other to a common shape
    input, other = torch.broadcast_tensors(input, other)

    # Determine the output tensor
    if out is None:
        out = torch.empty_like(input)

    # Calculate number of elements
    num_elements = input.numel()

    # Launch the Triton kernel
    grid = lambda meta: (triton.cdiv(num_elements, meta['BLOCK_SIZE']),)
    add_mean_kernel[grid](
        input_ptr=input,
        other_ptr=other,
        output_ptr=out,
        alpha=alpha,
        num_elements=num_elements,
        input_stride=input.stride(0),
        other_stride=other.stride(0),
        output_stride=out.stride(0),
        BLOCK_SIZE=1024  # Define block size
    )

    # Compute the mean along the specified dimension
    if dim is None:
        mean_result = out.mean()
    else:
        mean_result = out.mean(dim=dim, keepdim=keepdim)

    return mean_result

# Example usage:
input_tensor = torch.tensor([1.0, 2.0, 3.0])
other_tensor = torch.tensor([0.5, 0.5, 0.5])
result = add_mean(input_tensor, other_tensor, alpha=2)
print(result)
