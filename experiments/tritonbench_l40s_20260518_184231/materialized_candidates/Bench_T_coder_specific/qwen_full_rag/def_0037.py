import torch
import triton
import triton.language as tl


@triton.jit
def _fused_cosine_embedding_loss_with_normalization_kernel(
        input1, input2, target, N,
        loss_out, cos_out,
        margin, reduction,
        BLOCK_SIZE: tl.constexpr):
    """Triton kernel for computing cosine embedding loss."""
    # Get row indices
    pid = tl.program_id(axis=0)
    row_start = tl.program_id(axis=0) * BLOCK_SIZE
    row_indices = row_start + tl.arange(0, BLOCK_SIZE)
    row_mask = row_indices < N

    # Load data
    input1 = tl.load(input1 + row_indices)
    input2 = tl.load(input2 + row_indices)
    target = tl.load(target + row_indices).to(tl.float32)
    cos = tl.dot(input1, input2)

    # Normalize inputs
    input1_norm = tl.sqrt(tl.sum(input1 * input1))
    input2_norm = tl.sqrt(tl.sum(input2 * input2))
    input1 /= input1_norm
    input2 /= input2_norm

    # Compute loss
    cos = tl.where(row_mask, cos, 0.)
    target = tl.where(row_mask, target, 0.)
    loss = tl.where(target > 0, 1 - cos, tl.maximum(cos - margin, 0))

    # Apply reduction
    if reduction == 'mean':
        loss = tl.sum(loss) / N
        cos = tl.sum(cos) / N
    elif reduction == 'sum':
        pass
    else:
        loss = tl.where(row_mask, loss, 0.)
        cos = tl.where(row_mask, cos, 0.)

    # Store results
    tl.store(loss_out + row_indices, loss)
    tl.store(cos_out + row_indices, cos)


def fused_cosine_embedding_loss_with_normalization(input1: torch.Tensor, input2: torch.Tensor,
                                                   target: torch.Tensor, margin: float = 0,
                                                   reduction: str = 'mean') -> torch.Tensor:
    """Computes the cosine embedding loss between two normalized tensors."""

    # Validate arguments
    assert input1.shape == input2.shape
    N = input1.numel()
    input_shape = input1.shape
    input1 = input1.view(-1, input1.shape[-1])
    input2 = input2.view(-1, input2.shape[-1])
    target = target.view(-1)

    # Prepare output
    loss = torch.empty_like(target, dtype=torch.float32, device=target.device)
    cos = torch.empty_like(target, dtype=torch.float32, device=target.device)

    # Work out some constants
    BLOCK_SIZE = triton.next_power_of_2(input1.shape[1])

    # Less than 64KB per feature: enqueue fused kernel; otherwise fallback to slow path
    MAX_FUSED_SIZE = 65536 // input1.shape[-1]
    if BLOCK_SIZE <= MAX_FUSED_SIZE:
        num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
        _fused_cosine_embedding_loss_with_normalization_kernel[(N, )](
            input1,
            input2,
            target,
            N,
            loss,
            cos,
            margin,
            reduction,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
    else:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")

    # Reshape output
    loss = loss.reshape(input_shape)
    cos = cos.reshape(input_shape)

    return loss, cos
