import torch
import triton
import triton.language as tl

@triton.jit
def _fused_cosine_embedding_loss_kernel(
    in1_ptr, in2_ptr, tgt_ptr, out_ptr,
    margin, N, D,
    stride_in1, stride_in2, stride_tgt, stride_out,
    BLOCK_SIZE: tl.constexpr
):
    # Each program handles one row of the input tensors
    row_id = tl.program_id(0)
    # If out of range, skip
    if row_id >= N:
        return

    # ------------------------------------
    # Pass 1: Compute L2 norms for in1 and in2
    # ------------------------------------
    # partial sums for norms
    sum_sq_x = tl.float32(0.)
    sum_sq_y = tl.float32(0.)

    col_offs = tl.arange(0, BLOCK_SIZE)
    # base pointers for row
    in1_row_ptr = in1_ptr + row_id * stride_in1
    in2_row_ptr = in2_ptr + row_id * stride_in2

    # Loop over columns in blocks of BLOCK_SIZE
    for start_col in range(0, D, BLOCK_SIZE):
        cols = start_col + col_offs
        mask = cols < D

        x = tl.load(in1_row_ptr + cols, mask=mask, other=0.)
        y = tl.load(in2_row_ptr + cols, mask=mask, other=0.)

        sum_sq_x += tl.sum(x * x, where=mask)
        sum_sq_y += tl.sum(y * y, where=mask)

    norm_x = tl.sqrt(sum_sq_x)
    norm_y = tl.sqrt(sum_sq_y)

    # ------------------------------------
    # Pass 2: Compute the cosine similarity
    # ------------------------------------
    dot = tl.float32(0.)
    for start_col in range(0, D, BLOCK_SIZE):
        cols = start_col + col_offs
        mask = cols < D

        x = tl.load(in1_row_ptr + cols, mask=mask, other=0.)
        y = tl.load(in2_row_ptr + cols, mask=mask, other=0.)

        # Normalize per-row along dimension=1
        x = x / (norm_x + 1e-8)
        y = y / (norm_y + 1e-8)

        dot += tl.sum(x * y, where=mask)

    # ------------------------------------
    # Compute per-row loss
    # ------------------------------------
    t = tl.load(tgt_ptr + row_id * stride_tgt)
    # t should be 1 or -1
    is_pos = t > 0
    # For target = 1 => loss = 1 - cos
    # For target = -1 => loss = max(0, cos - margin)
    loss_pos = 1. - dot
    loss_neg = tl.maximum(tl.float32(0.), dot - margin)
    loss_val = tl.where(is_pos, loss_pos, loss_neg)

    # Store result
    tl.store(out_ptr + row_id * stride_out, loss_val)


def fused_cosine_embedding_loss_with_normalization(
    input1: torch.Tensor,
    input2: torch.Tensor,
    target: torch.Tensor,
    margin: float = 0.0,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Computes cosine embedding loss between two normalized tensors along dimension=1.
    """
    # Check input shapes
    assert input1.shape == input2.shape, "input1 and input2 must have the same shape"
    assert target.dim() == 1, "target should be a 1D tensor of shape [N]"
    assert target.shape[0] == input1.shape[0], "
