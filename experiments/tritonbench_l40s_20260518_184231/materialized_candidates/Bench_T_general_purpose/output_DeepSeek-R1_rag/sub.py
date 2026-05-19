import triton
import triton.language as tl
import torch
import math

@triton.jit
def sub_kernel_tensor(
    input_ptr,
    other_ptr,
    alpha,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    input_val = tl.load(input_ptr + offsets, mask=mask)
    other_val = tl.load(other_ptr + offsets, mask=mask)
    output_val = input_val - alpha * other_val
    tl.store(output_ptr + offsets, output_val, mask=mask)

@triton.jit
def sub_kernel_scalar(
    input_ptr,
    other_scalar,
    alpha,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    input_val = tl.load(input_ptr + offsets, mask=mask)
    other_val = other_scalar  # Scalar is broadcasted
    output_val = input_val - alpha * other_val
    tl.store(output_ptr + offsets, output_val, mask=mask)

def sub(input, other, *, alpha=1, out=None):
    # Compute the result type considering input, other, and alpha
    dtype = torch.result_type(input, other)
    alpha_tensor = torch.tensor(alpha, dtype=dtype)
    dtype = torch.result_type(dtype, alpha_tensor)

    # Handle broadcasting for tensors
    if isinstance(other, torch.Tensor):
        # Compute broadcasted shape
        try:
            broadcasted_shape = torch.broadcast_shapes(input.shape, other.shape)
        except Exception as e:
            raise ValueError(f"Shapes {input.shape} and {other.shape} are not broadcastable") from e
        input_expanded = input.expand(broadcasted_shape).to(dtype).contiguous()
        other_expanded = other.expand(broadcasted_shape).to(dtype).contiguous()
        n_elements = input_expanded.numel()
    else:
        # Convert other to a tensor for type promotion, then back to scalar
        other_tensor = torch.tensor(other, dtype=dtype, device=input.device)
        dtype = torch.result_type(input, other_tensor)
        input_expanded = input.to(dtype).contiguous()
        other = other_tensor.item() if other_tensor.numel() == 1 else other_tensor
        n_elements = input_expanded.numel()

    # Cast alpha to the appropriate type
    alpha = torch.tensor(alpha, dtype=dtype).item()

    # Handle output tensor
    if out is None:
        out = torch.empty_like(input_expanded, dtype=dtype)
    else:
        expected_shape = input_expanded.shape if isinstance(other, torch.Tensor) else input.shape
        if out.shape != expected_shape:
            raise ValueError(f"Output shape {out.shape} does not match expected shape {expected_shape}")
        if out.dtype != dtype:
            raise TypeError(f"Output dtype {out.dtype} does not match expected dtype {dtype}")
        out = out.contiguous()

    # Choose kernel and launch parameters
    BLOCK_SIZE = triton.next_power_of_2(1024)
    grid_size = triton.cdiv(n_elements, BLOCK_SIZE)

    if isinstance(other, torch.Tensor):
        sub_kernel_tensor[(grid_size,)](
            input_expanded, other_expanded, alpha, out, n_elements, BLOCK_SIZE
        )
    else:
        sub_kernel_scalar[(grid_size,)](
            input_expanded, other, alpha, out, n_elements, BLOCK_SIZE
        )

    return out
