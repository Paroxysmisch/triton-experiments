import torch
import triton
import triton.language as tl

@triton.jit
def _cosine_embedding_loss_kernel(
    input1_ptr, input2_ptr, target_ptr, output_ptr,
    stride_i1, stride_i2, stride_t,
    N, D, margin,
    BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr
):
    pid_m = tl.program_id(0)
    row_start = pid_m * BLOCK_M

    # Each program handles one row at a time in this simple scheme
    # If row_start >= N, we are out of range
    if row_start >= N:
        return

    # We'll compute the L2 norm of row of input1 and input2, then compute dot product
    row_norm1 = tl.float32(0.)
    row_norm2 = tl.float32(0.)
    row_dot   = tl.float32(0.)

    # Calculate row-wise partial for normalization and dot
    # We iterate over columns in chunks of BLOCK_D
    for col_block_start in range(0, D, BLOCK_D):
        offs = col_block_start + tl.arange(0, BLOCK_D)
        mask = offs < D

        # Load from input1, input2
        i1 = tl.load(input1_ptr + row_start * stride_i1 + offs, mask=mask, other=0.).to(tl.float32)
        i2 = tl.load(input2_ptr + row_start * stride_i2 + offs, mask=mask, other=0.).to(tl.float32)

        row_norm1 += tl.sum(i1 * i1, axis=0)
        row_norm2 += tl.sum(i2 * i2, axis=0)

    # Compute final norms
    norm1 = tl.sqrt(row_norm1)
    norm2 = tl.sqrt(row_norm2)

    # Now compute dot product with normalized inputs
    for col_block_start in range(0, D, BLOCK_D):
        offs = col_block_start + tl.arange(0, BLOCK_D)
        mask = offs < D

        i1 = tl.load(input1_ptr + row_start * stride_i1 + offs, mask=mask, other=0.).to(tl.float32)
        i2 = tl.load(input2_ptr + row_start * stride_i2 + offs, mask=mask, other=0.).to(tl.float32)

        # Normalize
        i1_normed = tl.where(norm1 > 0, i1 / norm1, 0.)
        i2_normed = tl.where(norm2 > 0, i2 / norm2, 0.)
        row_dot   += tl.sum(i1_normed * i2_normed, axis=0)

    # Load target
    t = tl.load(target_ptr + row_start * stride_t).to(tl.float32)

    # Compute per-element loss
    # if target == 1 => loss = 1 - cos
    # if target == -1 => loss = max(0, cos - margin)
    # cos == row_dot
    loss = tl.where(t > 0, 1. - row_dot, tl.maximum(tl.float32(0.), row_dot - margin))

    # Store
    tl.store(output_ptr + row_start, loss)


def fused_cosine_embedding_loss_with_normalization(
    input1: torch.Tensor,
    input2: torch.Tensor,
    target: torch.Tensor,
    margin: float = 0,
    reduction: str = 'mean'
) -> torch.Tensor:
    assert input1.shape == input2.shape, "input1 and input2 must have the same shape"
    assert input1.dim() == 2, "This implementation expects 2D tensors"
    assert target.dim() == 1 and target.shape[0] == input1.shape[0], "target must be 1D with matching batch size"
    assert reduction in ['none', 'mean', 'sum'], "reduction must be 'none', 'mean', or 'sum'"

    N, D = input1.shape
    # Allocate output
    out = torch.empty(N, device=input1.device, dtype=torch.float32)

    grid = lambda META: ( (N + META['BLOCK_M'] - 1) // META['BLOCK_M'], )

    _cosine_embedding_loss_kernel[grid](
        input1,
        input2,
        target,
        out,
        input1.stride(0),
        input2.stride(0),
        target.stride(0),
        N,
        D,
        margin,
        BLOCK_M=1,
        BLOCK_D=128
    )

    if reduction == 'none':
        return out
    elif reduction == 'sum':
        return out.sum()
    else:  # 'mean'
        return out.mean()
