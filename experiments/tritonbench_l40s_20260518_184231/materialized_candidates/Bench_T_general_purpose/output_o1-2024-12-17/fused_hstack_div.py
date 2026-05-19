import torch
import triton
import triton.language as tl


@triton.jit
def _read_input_value_4(
    idx,
    in_ptr0, offset0,
    in_ptr1, offset1,
    in_ptr2, offset2,
    in_ptr3, offset3,
    n_tensors
):
    # Initialize result
    val = tl.zeros([], dtype=tl.float32)

    # For each tensor that actually exists, select the correct pointer when idx is in range
    # Tensor 0
    if n_tensors > 0:
        in_range0 = (idx >= offset0) & (idx < offset1)
        val0 = tl.where(in_range0, tl.load(in_ptr0 + (idx - offset0), mask=in_range0), 0.0)
        val += val0

    # Tensor 1
    if n_tensors > 1:
        in_range1 = (idx >= offset1) & (idx < offset2)
        val1 = tl.where(in_range1, tl.load(in_ptr1 + (idx - offset1), mask=in_range1), 0.0)
        val += val1

    # Tensor 2
    if n_tensors > 2:
        in_range2 = (idx >= offset2) & (idx < offset3)
        val2 = tl.where(in_range2, tl.load(in_ptr2 + (idx - offset2), mask=in_range2), 0.0)
        val += val2

    # Tensor 3
    #
