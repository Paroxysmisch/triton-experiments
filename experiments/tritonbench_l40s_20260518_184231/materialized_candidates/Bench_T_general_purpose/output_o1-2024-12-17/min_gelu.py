import triton
import triton.language as tl
import torch
from collections import namedtuple

MinGeluResult = namedtuple("MinGeluResult", ["values", "indices"])


@triton.jit
def _gelu_kernel(
    x_ptr,  # pointer to input
    y_ptr,  # pointer to output (GELU of input)
    n_elements,
    approx: tl.constexpr,  # 'none' or 'tanh'
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    # Compute the offset for this program
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    if approx == 'tanh':
        # approximate GELU: 0.5 * x * (1 + Tanh( sqrt(2/pi) * (x + 0.044715*x^3) ))
        c = 0.044715
        sqrt_2_over_pi = 0.7978845608  # approx sqrt(2/pi)
        inner = sqrt_2_over_pi * (x + c * x * x * x)
        gelu_x = 0.5 * x * (1.0 + tl.tanh(inner))
    else:
        # exact GELU: x * Φ(x) ~ x * 0.5[1 + erf(x / sqrt(2))]
        # but we can approximate erf via math library or built-in
        # for demonstration: use built-in `tl.erf`
        one_over_sqrt2 = 0.7071067812
        gelu_x = x * 0.5 * (1.0 + tl.erf(x * one_over_sqrt2))

    tl.store(y_ptr + offsets, gelu_x, mask=mask)


@triton.jit
def _min_reduce_1d_kernel(
    x_ptr,    # pointer to the (already GELU'ed) input
    idx_ptr,  # pointer to the output indices
    val_ptr,  # pointer to the output values
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    """
    Reduce 1D data in parallel to find the global minimum value and index.
    This kernel uses one block. For large n_elements, multiple reductions
    or a multi-block approach would be required. Demonstration only.
    """
    # We'll assume a single block for simplicity of demonstration.
    # Load elements in a thread-local array, reduce in shared memory.
    offsets = tl.arange(0, BLOCK_SIZE)
    # Initialize local buffer
    min_val = tl.full([BLOCK_SIZE], float('inf'), dtype=tl.float32)
    min_idx = tl.full([BLOCK_SIZE], 0, dtype=tl.int32)

    # If n_elements > BLOCK_SIZE, you could iterate in steps of BLOCK_SIZE here.
    # For brevity, we do a single pass assuming n_elements <= BLOCK_SIZE.
    # For real usage, you'd do a multi-pass or multi-block approach.

    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=float('inf'))
    min_val = tl.where(x < min_val, x, min_val)
    min_idx = tl.where(x < min_val, offsets, min_idx)

    # Parallel reduction inside the warp/block
    # step = BLOCK_SIZE // 2, etc.
    step = BLOCK_SIZE // 2
    while step > 0:
        lhs_val = min_val[:step]
        rhs_val = min_val[step:step*2]
        lhs_idx = min_idx[:step]
        rhs_idx = min_idx[step:step*2]

        # Compare
        cond = rhs_val < lhs_val
        new_val = tl.where(cond, rhs_val, lhs_val)
        new_idx = tl.where(cond, rhs_idx, lhs_idx)

        min_val = tl.concatenate([new_val, min_val[step*2:]], 0)
        min_idx = tl.concatenate([new_idx, min_idx[step*2:]], 0)

        step //= 2

    # By the end, min_val[0] and min_idx[0] are the global min and its index
    if 0 == tl.program_id(0):
        tl.store(val_ptr, min_val[0])
        tl.store(idx_ptr, min_idx[0])


def min_gelu(input, dim=None, keepdim=False, approximate='none', out=None):
    """
    min_gelu(input, dim=None, keepdim=False, approximate='none', out=None) -> Tensor

    Computes the GELU activation (exact or approximate) on 'input',
    then returns the minimum value along 'dim', or the global minimum
    if 'dim' is None.

    If 'dim' is specified, returns a namedtuple (values, indices).
    Otherwise, returns the minimum value tensor. 'keepdim' controls
    whether the dimension is kept as size=1. 'approximate' controls
    the GELU formula used. If 'out' is provided, places the result
    into 'out' (for the values). 
    Indices are not placed into 'out'.
    """
    # Basic checks
    if not isinstance(input, torch.Tensor):
        raise TypeError("input must be a torch.Tensor")

    if approximate not in ('none', 'tanh'):
        raise ValueError("approximate must be either 'none' or 'tanh'")

    # Step 1: Compute GELU via a Triton kernel elementwise.
    x = input.contiguous()
    n_elements = x.numel
