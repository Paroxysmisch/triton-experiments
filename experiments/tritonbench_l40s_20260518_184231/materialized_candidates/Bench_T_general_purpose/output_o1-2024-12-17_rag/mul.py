import triton
import triton.language as tl
import torch
import math

@triton.jit
def _mul_kernel_tensor(
    A_ptr, B_ptr, C_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    a_vals = tl.load(A_ptr + offsets, mask=mask)
    b_vals = tl.load(B_ptr + offsets, mask=mask)
    c_vals = a_vals * b_vals
    tl.store(C_ptr + offsets, c_vals, mask=mask)

@triton.jit
def _mul_kernel_scalar(
    A_ptr, scalar: tl.constexpr, C_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    a_vals = tl.load(A_ptr + offsets, mask=mask)
    c_vals = a_vals * scalar
    tl.store(C_ptr + offsets, c_vals, mask=mask)

def mul(input, other, *, out=None):
    result_type = torch.result_type(input, other)
    if isinstance(other, torch.Tensor):
        # Determine broadcast shape
        broadcast_shape = torch.broadcast_shapes(input.shape, other.shape)
        input_ = input.to(result_type).expand(broadcast_shape).contiguous()
        other_ = other.to(result_type).expand(broadcast_shape).contiguous()
        if out is None:
            out = torch.empty(broadcast_shape, dtype=result_type, device=input.device)
        else:
            if out.shape != broadcast_shape:
                raise ValueError("Output shape does not match broadcast shape.")
            if out.dtype != result_type:
                raise ValueError("Output dtype does not match result type.")
        n_elements = out.numel()
        block_size = triton.next_power_of_2(min(n_elements, 1024))
        grid_size = (n_elements + block_size - 1) // block_size
        _mul_kernel_tensor[(grid_size,)](
            input_, other_, out,
            n_elements,
            block_size
        )
    else:
        # Scalar multiplication
        input_ = input.to(result_type).contiguous()
        if out is None:
            out = torch.empty_like(input_, dtype=result_type)
        else:
            if out.shape != input_.shape:
                raise ValueError("Output shape does not match input tensor shape.")
            if out.dtype != result_type:
                raise ValueError("Output dtype does not match result type.")
        n_elements = out.numel()
        block_size = triton.next_power_of_2(min(n_elements, 1024))
        grid_size = (n_elements + block_size - 1) // block_size
        if isinstance(other, complex):
            # For complex scalars, handle real and imaginary parts via Python
            real_part = other.real
            imag_part = other.imag
            # Multiply in Python, store in out for each element
            temp = input_.cpu().numpy() * complex(real_part, imag_part)
            out.copy_(torch.from_numpy(temp).to(out.device))
        else:
            _mul_kernel_scalar[(grid_size,)](
                input_,
                other,
                out,
                n_elements,
                block_size
            )
    return out
