import triton
import triton.language as tl
import torch

@triton.jit
def _fused_repeat_interleave_log_softmax_kernel(
    input_ptr, output_ptr,
    repeats_ptr, prefix_sum_ptr,
    B, N, M,
    BLOCK_SIZE: tl.constexpr
):
    row_id = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    base_index = row_id * M + col_offsets

    # Load integer repeats (if a single int is provided, it is broadcasted on the Python side)
    # prefix_sum_ptr holds the prefix sums of repeats for quick lookup.
    # The repeated index for each position is found by binary searching "col_offsets" in prefix_sum.
    # A simple linear search can be done for demonstration (inefficient for large sizes).

    # Load repeats in register for row, if repeats varies per element, it must be computed from prefix_sum.
    # For demonstration, handle either single int or per-element repeats via prefix_sum search.
    # If single int, just do direct integer division. Otherwise, do prefix_sum-based indexing.

    # If we assume a single int repeat, skip the prefix sum:
    #   source_col = col_offsets // repeats
    # If not, linear scan on prefix_sum to find correct source_col.
    repeats_val = tl.load(repeats_ptr)  # reading first element to check
    is_uniform = tl.arange(0, 1) < 0  # dummy, we will branch in Python wrapper if needed

    # Example naive approach for per-element repeats:
    # We'll linearly search prefix_sum for each col to find which input index to gather from.
    # This is slow for large arrays but demonstrates the method.
    def find_source_col(pos):
        # linear scan
        source_idx = 0
        current_prefix = tl.load(prefix_sum_ptr + source_idx)
        while (pos >= current_prefix) and (source_idx < N - 1):
            source_idx += 1
            current_prefix = tl.load(prefix_sum_ptr + source_idx)
        return source_idx

    col_in = tl.zeros_like(col_offsets)
    if tl.load(repeats_ptr) == tl.load(repeats_ptr + 0):
        # Single int repeats case (check if the first repeat equals them all, done in Python)
        col_in = col_offsets // repeats_val
    else:
        # Per-element repeat case
        col_in_tmp = [find_source_col(pos) for pos in col_offsets]
        col_in = tl.stack(col_in_tmp)

    mask = base_index < (row_id + 1) * M
    src_index = row_id * N + col_in
    x = tl.where(
        mask,
        tl.load(input_ptr + src_index, mask=mask),
        float("-inf")
    )

    # Compute log-softmax along the last dimension for this row (assuming M <= BLOCK_SIZE)
    x_max = tl.max(x, axis=0)
    x_shifted = x - x_max
    x_exp = tl.exp(x_shifted)
    exp_sum = tl.sum(x_exp, axis=0)
    out_val = x_shifted - tl.log(exp_sum)

    tl.store(output_ptr + base_index, out_val, mask=mask)


def fused_repeat_interleave_log_softmax(input, repeats, dim=None, *, output_size=None, dtype=None, out=None):
    """
    fused_repeat_interleave_log_softmax(input, repeats, dim=None, *, output_size=None, dtype=None, out=None) -> Tensor
    """
    # Convert input to contiguous float tensor
    x = input if input.is_contiguous() else input.contiguous()
    if dtype is not None:
        x = x.to(dtype)
    original_shape = x.shape

    # If dim is None, flatten input
    if dim is None:
        x = x.flatten()
        dim = 0

    # Handle negative dims
    if dim < 0:
        dim += x.dim()

    # Compute repeated size
    # If repeats is an int, apply uniformly
    if isinstance(repeats, int):
        # Expand along dim
        out_shape = list(x.shape)
        out_shape[dim] = out_shape[dim] * repeats
        repeated_size = tuple(out_shape)
        # Prepare output
        if out is not None:
            out_t = out
        else:
            out_t = x.new_empty(repeated_size)
        # For simplicity, handle only dim = last dimension in Triton example
        if dim != x.dim() - 1:
            # Fallback to PyTorch for other dims (demonstration only)
            repeated = x.repeat_interleave(repeats, dim=dim)
            out_t_pyt = torch.log_softmax(repeated, dim=dim)
            out_t.copy_(out_t_pyt)
            return out_t
        B, N = x.shape[-2], x.shape[-1]  # assume 2D for demonstration, last dim repeated
        M = N * repeats

        # Make pointers
        input_ptr = x.contiguous().data_ptr()
        output_ptr = out_t.data_ptr()
        # Create a tensor for repeats on device
        repeats_tensor = torch.tensor([
