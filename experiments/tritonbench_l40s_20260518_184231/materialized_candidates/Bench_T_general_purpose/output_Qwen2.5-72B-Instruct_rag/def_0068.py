import torch
import triton
import triton.language as tl

@triton.jit
def add_mean_kernel(
    input_ptr, other_ptr, output_ptr, alpha, stride_input, stride_other, stride_output,
    input_size, other_size, output_size, dim, keepdim, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    offsets_input = block_start + tl.arange(0, BLOCK_SIZE)
    offsets_other = block_start + tl.arange(0, BLOCK_SIZE)
    offsets_output = block_start + tl.arange(0, BLOCK_SIZE)

    mask_input = offsets_input < input_size
    mask_other = offsets_other < other_size
    mask_output = offsets_output < output_size

    input_vals = tl.load(input_ptr + offsets_input * stride_input, mask=mask_input, other=0.0)
    other_vals = tl.load(other_ptr + offsets_other * stride_other, mask=mask_other, other=0.0)

    result = input_vals + alpha * other_vals

    if dim is not None:
        # Compute mean along the specified dimension
        result = tl.sum(result, axis=dim) / result.shape[dim]
        if keepdim:
            result = tl.reshape(result, result.shape[:dim] + (1,) + result.shape[dim+1:])

    tl.store(output_ptr + offsets_output * stride_output, result, mask=mask_output)

def add_mean(input, other, dim=None, alpha=1, keepdim=False, dtype=None, out=None) -> torch.Tensor:
    # Handle dtype conversion
    if dtype is not None:
        input = input.to(dtype)
        other = other.to(dtype)

    # Handle broadcasting
    input, other = torch.broadcast_tensors(input, other)

    # Compute the shape of the output tensor
    if dim is None:
        output_shape = (1,)
    else:
        output_shape = list(input.shape)
        if not keepdim:
            output_shape[dim] = 1

    # Allocate output tensor if not provided
    if out is None:
        out = torch.empty(output_shape, dtype=input.dtype, device=input.device)
    else:
        assert out.shape == output_shape, "Output tensor shape mismatch"

    # Determine the number of elements to process
    input_size = input.numel()
    other_size = other.numel()
    output_size = out.numel()

    # Determine the strides
    stride_input = input.stride(0)
    stride_other = other.stride(0)
    stride_output = out.stride(0)

    # Launch the Triton kernel
    grid = lambda meta: (triton.cdiv(input_size, meta['BLOCK_SIZE']),)
    add_mean_kernel[grid](
        input_ptr=input.data_ptr(),
        other_ptr=other.data_ptr(),
        output_ptr=out.data_ptr(),
        alpha=alpha,
        stride_input=stride_input,
        stride_other=stride_other,
        stride_output=stride_output,
        input_size=input_size,
        other_size=other_size,
        output_size=output_size,
        dim=dim,
        keepdim=keepdim,
        BLOCK_SIZE=1024
    )

    return out

# Test case 1: Simple addition and mean over all elements
input = torch.tensor([1.0, 2.0, 3.0], device='cuda')
other = torch.tensor([1.0, 1.0, 1.0], device='cuda')
result = add_mean(input, other, alpha=2)
print(result)  # Expected: 4.0

# Test case 2: Mean along a specific dimension
input = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
other = torch.tensor([[1.0, 1.0], [1.0, 1.0]], device='cuda')
result = add_mean(input, other, dim=1)
print(result)  # Expected: [2.0, 4.0]

# Test case 3: Keep dimension
result = add_mean(input, other, dim=1, keepdim=True)
print(result)  # Expected: [[2.0], [4.0]]

# Test case 4: Type promotion
input = torch.tensor([1, 2, 3], device='cuda', dtype=torch.int32)
other = torch.tensor([1.0, 1.0, 1.0], device='cuda')
result = add_mean(input, other, alpha=2, dtype=torch.float32)
print(result)  # Expected: 4.0

# Test case 5: Broadcasting
input = torch.tensor([1.0, 2.0, 3.0], device='cuda')
other = torch.tensor(1.0, device='cuda')
result = add_mean(input, other, alpha=2)
print(result)  # Expected: 4.0
