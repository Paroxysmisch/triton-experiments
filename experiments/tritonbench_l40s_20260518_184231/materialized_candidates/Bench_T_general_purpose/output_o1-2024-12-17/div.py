import triton
import triton.language as tl
import torch

@triton.jit
def _div_kernel(
    input_ptr,
    other_ptr,
    out_ptr,
    n_elements,
    rounding_mode_val,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(other_ptr + offsets, mask=mask, other=1.0)
    
    # Default float division
    result = x / y

    # If truncation is indicated, truncate toward zero
    is_trunc = rounding_mode_val == 1
    # If floor is indicated, floor the result
    is_floor = rounding_mode_val == 2

    # Conditionally apply rounding
    # We'll cast to float here to ensure we can do floor/trunc
    float_res = result.to(tl.float32)
    trunc_res = tl.where(
        is_trunc,
        tl.cast(tl.math.trunc(float_res), tl.float32),
        float_res
    )
    floor_res = tl.where(
        is_floor,
        tl.cast(tl.math.floor(trunc_res), tl.float32),
        trunc_res
    )

    final_res = floor_res
    tl.store(out_ptr + offsets, final_res, mask=mask)


def div(input, other, *, rounding_mode=None, out=None):
    """
    div(input, other, *, rounding_mode=None, out=None) -> Tensor
    Performs element-wise division with optional rounding.
    Supports broadcasting, type promotion, and integer/float/complex inputs.
    By default, performs 'true' division. If rounding_mode is 'trunc',
    it truncates the result towards zero. If rounding_mode is 'floor',
    it ensures floor rounding.
    """
    # Handle inputs (promote to float if they are int)
    if isinstance(input, torch.Tensor):
        if input.dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
            input = input.to(torch.get_default_dtype())
    else:
        # Wrap scalars in tensor if needed
        input = torch.tensor(input, dtype=torch.get_default_dtype())

    if isinstance(other, torch.Tensor):
        if other.dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
            other = other.to(torch.get_default_dtype())
    else:
        other = torch.tensor(other, dtype=torch.get_default_dtype())

    # Broadcast shapes
    broadcast_shape = torch.broadcast_shapes(input.shape, other.shape)
    broadcasted_input = input.expand(broadcast_shape)
    broadcasted_other = other.expand(broadcast_shape)

    # Prepare output
    if out is None:
        out = torch.empty_like(broadcasted_input, dtype=torch.get_default_dtype())
    else:
        # Ensure out can broadcast to broadcast_shape
        if out.shape != broadcast_shape:
            raise ValueError("Output tensor shape must match broadcasted shape.")
        if out.dtype != torch.get_default_dtype():
            raise ValueError("Output dtype must match default scalar type.")

    n_elements = broadcasted_input.numel()

    # Convert rounding_mode to a numeric flag for Triton:
    # 0 = None, 1 = 'trunc', 2 = 'floor'
    mode_flag = 0
    if rounding_mode == 'trunc':
        mode_flag = 1
    elif rounding_mode == 'floor':
        mode_flag = 2

    # Launch kernel
    grid = lambda META: ((n_elements + META['BLOCK_SIZE'] - 1) // META['BLOCK_SIZE'],)

    _div_kernel[grid](
        broadcasted_input.contiguous().data_ptr(),
        broadcasted_other.contiguous().data_ptr(),
        out.contiguous().data_ptr(),
        n_elements,
        mode_flag,
        BLOCK_SIZE=1024
    )

    return out
