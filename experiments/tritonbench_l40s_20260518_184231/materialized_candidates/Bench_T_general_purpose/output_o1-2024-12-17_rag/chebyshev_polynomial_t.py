import math
import torch
import triton
import triton.language as tl

@triton.jit
def _chebyshev_kernel(input_ptr, n_ptr, out_ptr, size, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < size

    x = tl.load(input_ptr + offsets, mask=mask)
    n_val = tl.load(n_ptr)  # Assuming n is a 1-element tensor

    # Base cases
    # T0(x) = 1
    # T1(x) = x
    out_val = tl.zeros_like(x)
    is_n0 = n_val == 0
    is_n1 = n_val == 1
    out_val = tl.where(is_n0, 1.0, out_val)
    out_val = tl.where(is_n1 & ~is_n0, x, out_val)

    # We determine recursion vs. trig for each element based on n_val and abs(x)
    # T_{n+1}(x) = 2*x*T_{n}(x) - T_{n-1}(x)
    # T_n(x) = cos(n * arccos(x))
    use_recursion = (n_val < 6) | (tl.abs(x) > 1.0)

    # For recursion path
    # If 0,1 => already handled
    # Loop from k=2..n_val
    # T_{k+1}(x) = 2*x*T_{k}(x) - T_{k-1}(x)
    # We'll keep T_{k-1}(x) and T_{k}(x) in registers
    if tl.any(use_recursion):
        # T0 = 1, T1 = x
        t_km1 = 1.0
        t_k = x
        tmp_out = tl.zeros_like(x)
        # If n_val=0 or 1, t_k or t_km1 might already be the final, but we still compute full series
        # We'll do a loop up to n_val, storing final in tmp_out
        k = 2
        while k <= n_val:
            t_kp1 = 2.0 * x * t_k - t_km1
            t_km1 = t_k
            t_k = t_kp1
            k += 1
        tmp_out = tl.where(n_val == 0, 1.0, tmp_out)         # T0
        tmp_out = tl.where((n_val == 1) & (n_val != 0), x, tmp_out)  # T1
        # t_k has Tn after finishing the loop
        tmp_out = tl.where((n_val >= 2), t_k, tmp_out)

        # Merge recursion result only if we decided to use recursion
        out_val = tl.where(use_recursion, tmp_out, out_val)

    # For trig path
    # Tn(x) = cos(n*arccos(x)) for each element if not recursion
    if tl.any(~use_recursion):
        # arccos might be undefined if abs(x) > 1, but there's a check to only do trig if abs(x)<=1, n>=6
        safe_zone = (tl.abs(x) <= 1.0) & (n_val >= 6)
        trig_val = tl.cos(n_val * tl.acos(x))
        out_val = tl.where(safe_zone, trig_val, out_val)

    # Write result
    tl.store(out_ptr + offsets, out_val, mask=mask)


def chebyshev_polynomial_t(input: torch.Tensor, n: torch.Tensor, *, out: torch.Tensor = None) -> torch.Tensor:
    if out is None:
        out = torch.empty_like(input)

    if input.numel() == 0:
        return out

    # Ensure on same device
    assert input.is_cuda and n.is_cuda, "Tensors must be on CUDA device."
    assert out.is_cuda, "Output tensor must be on CUDA device."

    size = input.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: ( (size + BLOCK_SIZE - 1) // BLOCK_SIZE, )

    _chebyshev_kernel[grid](
        input_ptr = input.data_ptr(),
        n_ptr = n.data_ptr(),
        out_ptr = out.data_ptr(),
        size = size,
        BLOCK_SIZE = BLOCK_SIZE
    )
    return out.view(input.shape)
