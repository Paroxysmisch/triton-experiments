import triton
import triton.language as tl
import torch
import math

@triton.jit
def _fused_hstack_div_kernel(
    X_ptr, D_ptr, Y_ptr,
    n_elements,
    rounding_mode: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x_val = tl.load(X_ptr + offsets, mask=mask)
    d_val = tl.load(D_ptr + offsets, mask=mask)
    div_val = x_val / d_val

    # 0 -> None, 1 -> 'trunc', 2 -> 'floor'
    if rounding_mode == 0:
        res = div_val
    elif rounding_mode == 1:
        sign = tl.where(div_val >= 0, 1.0, -1.0)
        abs_val = tl.abs(div_val)
        truncated = tl.floor(abs_val)
        res = sign * truncated
    else:  # rounding_mode == 2
        res = tl.floor(div_val)

    tl.store(Y_ptr + offsets, res, mask=mask)

def fused_hstack_div(tensors, divisor, *, rounding_mode=None, out=None):
    X = torch.hstack(tensors)

    # If both X and divisor are integer tensors and rounding_mode is None, promote to default float
    if rounding_mode is None:
        if X.dtype in (torch.int8, torch.int16, torch.int32, torch.int64) and (
            (isinstance(divisor, torch.Tensor) and divisor.dtype in (torch.int8, torch.int16, torch.int32, torch.int64))
            or (not isinstance(divisor, torch.Tensor) and isinstance(divisor, int))
        ):
            default_dtype = torch.get_default_dtype()
            X = X.to(default_dtype)
            if isinstance(divisor, torch.Tensor):
                divisor = divisor.to(default_dtype)
            else:
                divisor = torch.tensor(divisor, dtype=default_dtype, device=X.device)

    if not isinstance(divisor, torch.Tensor):
        divisor = torch.tensor(divisor, dtype=X.dtype, device=X.device)
    else:
        divisor = divisor.to(X.device, X.dtype)

    divisor_expanded = divisor.expand_as(X)

    if out is None:
        Y = torch.empty_like(X, dtype=torch.float32 if rounding_mode is not None else X.dtype)
    else:
        Y = out

    # Rounding mode codes
    mode_map = {None: 0, 'trunc': 1, 'floor': 2}
    if rounding_mode not in mode_map:
        raise ValueError("Invalid rounding_mode. Must be None, 'trunc', or 'floor'.")
    rm_code = mode_map[rounding_mode]

    n_elements = X.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    _fused_hstack_div_kernel[grid](
        X, divisor_expanded, Y,
        n_elements,
        rm_code,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return Y
