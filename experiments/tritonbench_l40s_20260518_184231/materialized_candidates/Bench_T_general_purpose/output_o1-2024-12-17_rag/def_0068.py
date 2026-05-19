import torch
import triton
import triton.language as tl
from typing import Optional, Union, Tuple


@triton.jit
def _add_mean_fwd_kernel(
    ptr_input,        # input tensor
    ptr_other,        # other tensor (already broadcasted shape or scalar expanded)
    ptr_out,          # partial sums output
    alpha,            # scalar multiplier for other
    N: tl.constexpr,  # number of rows (leading dimension after possible reshape/transpose)
    D: tl.constexpr,  # dimension being reduced
    B: tl.constexpr,  # block size
    HAS_OTHER: tl.constexpr
):
    """
    Each program instance processes a block of size B along D for a specific row in [0..N).
    """
    row_id = tl.program_id(0)
    block_id = tl.program_id(1)

    # Starting index for this block along D
    offset = row_id * D + block_id * B
    # [0..B-1] relative indices for the block
    block_offsets = tl.arange(0, B)
    idxs = offset + block_offsets
    mask = idxs < (row_id * D + D)

    # Load input
    val_input = tl.load(ptr_input + idxs, mask=mask, other=0.0)

    # Conditionally load "other" and apply alpha
    if HAS_OTHER:
        val_other = tl.load(ptr_other + idxs, mask=mask, other=0.0)
        val_input = val_input + alpha * val_other
    else:
        # This path handles the case where other is effectively 0
        pass

    # Sum over the B elements in this block
    partial_sum = tl.sum(val_input, 0)

    # Store the partial sum for this block
    # The intermediate buffer has shape [N, ceil_div(D, B)]
    ND = (D + B - 1) // B
    tl.store(ptr_out + row_id * ND + block_id, partial_sum)


def add_mean(
    input: torch.Tensor,
    other: Union[torch.Tensor, complex, float, int],
    dim: Optional[Union[int, Tuple[int, ...]]] = None,
    alpha: Union[complex, float, int] = 1,
    keepdim: bool = False,
    dtype: Optional[torch.dtype] = None,
    out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Adds the other tensor, scaled by alpha, to the input tensor and computes the mean value
    along the specified dimension. If no dimension is specified, computes the mean over all elements.
    """
    # Ensure 'other' is a tensor for broadcasting and type promotion,
    # handle scalar -> Tensor conversion.
    if not isinstance(other, torch.Tensor):
        other = torch.tensor(other, dtype=input.dtype, device=input.device)

    # Handle optional dtype casting before operation (for input and other).
    # Type promotion logic between input and other
    common_dtype = torch.promote_types(input.dtype, other.dtype)
    if dtype is not
