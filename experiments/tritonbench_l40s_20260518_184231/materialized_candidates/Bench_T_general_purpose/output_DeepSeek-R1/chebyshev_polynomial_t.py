import torch
import triton
import triton.language as tl
from typing import Optional

@triton.jit
def chebyshev_polynomial_t_kernel(
    input_ptr,
    n_ptr,
    output_ptr,
    input_element_stride,
    n_element_stride,
    output_element_stride,
    size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < size

    # Load input and n values
    input_offset = offsets * input_element_stride
    n_offset = offsets * n_element_stride
    x = tl.load(input_ptr + input_offset, mask=mask)
    n_val = tl.load(n_ptr + n_offset, mask=mask).to(tl.int32)

    # Initialize result
    result = tl.zeros_like(x)

    # Case n == 0
    mask_n0 = (n_val == 0)
    result = tl.where(mask_n0, 1.0, result)

    # Case n == 1
    mask_n1 = (n_val == 1)
    result = tl.where(mask_n1, x, result)

    # Case n >= 2
    mask_nge2 = (n_val >= 2)
    abs_x = tl.abs(x)
    use_recurrence = mask_nge2 & ((n_val < 6) | (abs_x > 1))
    use_trig = mask_nge2 & ~use_recurrence

    # Handle recurrence cases
    t0 = tl.full(x.shape, 1.0, dtype=x.dtype)
    t1 = x
    current_n = 2
    # Loop up to n_val (assuming n_val is not too large for practical unrolling)
    for _ in range(5):  # Covers n_val up to 6 (since n_val >=2, max iterations needed is 5)
        # Update only elements where current_n <= n_val and use_recurrence
        update_mask = (current_n <= n_val) & use_recurrence
        t_next = 2.0 * x * t1 - t0
        t0 = tl.where(update_mask, t1, t0)
        t1 = tl.where(update_mask, t_next, t1)
        current_n += 1
    recurrence_result = tl.where(use_recurrence, t1, 0.0)
    result = tl.where(use_recurrence, recurrence_result, result)

    # Handle trigonometric cases
    x_trig = tl.where(abs_x > 1.0, 1.0, x)  # Clamp to avoid NaN, though |x| <=1 is ensured by use_trig
    angle = tl.math.acos(x_trig)
    trig_arg = n_val.to(tl.float32) * angle
    trig_result = tl.math.cos(trig_arg)
    result = tl.where(use_trig, trig_result, result)

    # Store result
    output_offset = offsets * output_element_stride
    tl.store(output_ptr + output_offset, result, mask=mask)

def chebyshev_polynomial_t(input: torch.Tensor, n: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    # Broadcast input and n to the same shape
    broadcast_shape = torch.broadcast_shapes(input.shape, n.shape)
    input_broadcast = input.expand(broadcast_shape)
    n_broadcast = n.expand(broadcast_shape).to(torch.int32)
    
    # Allocate output tensor
    if out is None:
        out = torch.empty_like(input_broadcast)
    else:
        if out.shape != broadcast_shape:
            raise ValueError("Output tensor shape does not match broadcast shape")
        if not out.is_contiguous():
            raise ValueError("Output tensor must be contiguous")
    
    # Flatten tensors for kernel launch
    size = out.numel()
    input_flat = input_broadcast.view(-1)
    n_flat = n_broadcast.view(-1)
    out_flat = out.view(-1)
    
    # Define kernel launch parameters
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(size, meta['BLOCK_SIZE']),)
    
    # Launch kernel
    chebyshev_polynomial_t_kernel[grid](
        input_flat, n_flat, out_flat,
        input_flat.stride(0),
        n_flat.stride(0),
        out_flat.stride(0),
        size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out
