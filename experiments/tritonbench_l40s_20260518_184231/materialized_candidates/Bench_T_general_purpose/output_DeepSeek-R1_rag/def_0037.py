import torch
import triton
import triton.language as tl

@triton.jit
def cosine_embedding_loss_kernel(
    input1_ptr, input2_ptr, target_ptr, output_ptr,
    n_rows, d_cols,
    margin, reduction: tl.constexpr,
    input1_row_stride, input1_col_stride,
    input2_row_stride, input2_col_stride,
    target_row_stride,
    output_row_stride,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    if row_idx >= n_rows:
        return

    # Load target value for current row
    target_offset = row_idx * target_row_stride
    target_val = tl.load(target_ptr + target_offset).to(tl.float32)

    # Initialize accumulators for L2 norms and dot product
    sum_input1_sq = 0.0
    sum_input2_sq = 0.0
    dot_product = 0.0

    # Iterate over columns in blocks
    for col_idx in range(0, d_cols, BLOCK_SIZE):
        cols = col_idx + tl.arange(0, BLOCK_SIZE)
        mask = cols < d_cols

        # Load input1 and input2 values for current block
        input1_offset = row_idx * input1_row_stride + cols * input1_col_stride
        input1_val = tl.load(input1_ptr + input1_offset, mask=mask, other=0.0).to(tl.float32)
        input2_offset = row_idx * input2_row_stride + cols * input2_col_stride
        input2_val = tl.load(input2_ptr + input2_offset, mask=mask, other=0.0).to(tl.float32)

        # Update accumulators
        sum_input1_sq += tl.sum(input1_val * input1_val)
        sum_input2_sq += tl.sum(input2_val * input2_val)
        dot_product += tl.sum(input1_val * input2_val)

    # Compute norms and cosine similarity
    norm1 = tl.sqrt(sum_input1_sq)
    norm2 = tl.sqrt(sum_input2_sq)
    cosine_sim = dot_product / (norm1 * norm2 + 1e-8)

    # Calculate loss based on target
    if target_val == 1.0:
        loss = 1.0 - cosine_sim
    else:
        loss = tl.maximum(cosine_sim - margin, 0.0)

    # Store or accumulate loss based on reduction
    if reduction == 'none':
        output_offset = row_idx * output_row_stride
        tl.store(output_ptr + output_offset, loss)
    else:
        tl.atomic_add(output_ptr, loss)

def fused_cosine_embedding_loss_with_normalization(
    input1: torch.Tensor, input2: torch.Tensor, target: torch.Tensor,
    margin: float = 0, reduction: str = 'mean'
) -> torch.Tensor:
    assert input1.dim() == 2 and input2.dim() == 2, "Inputs must be 2D"
    assert target.dim() == 1, "Target must be 1D"
    assert input1.size(0) == input2.size(0) == target.size(0), "Batch size mismatch"
    assert input1.size(1) == input2.size(1), "Feature dimension mismatch"
    assert reduction in ['none', 'mean', 'sum'], "Invalid reduction"

    n_rows = input1.size(0)
    d_cols = input1.size(1)

    if reduction == 'none':
        output = torch.empty(n_rows, device=input1.device, dtype=torch.float32)
    else:
        output = torch.zeros(1, device=input1.device, dtype=torch.float32)

    BLOCK_SIZE = 1024  # Tune based on GPU capabilities

    grid = (n_rows,)
    cosine_embedding_loss_kernel[grid](
        input1, input2, target, output,
        n_rows, d_cols,
        margin, reduction,
        input1.stride(0), input1.stride(1),
        input2.stride(0), input2.stride(1),
        target.stride(0),
        output.stride(0) if reduction == 'none' else 0,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    if reduction == 'mean':
        output /= n_rows

    return output if reduction != 'none' else output
