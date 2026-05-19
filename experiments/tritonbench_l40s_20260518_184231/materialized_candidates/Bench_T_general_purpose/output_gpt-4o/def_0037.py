import torch
import triton
import triton.language as tl

@triton.jit
def cosine_embedding_loss_kernel(
    input1_ptr, input2_ptr, target_ptr, output_ptr,
    n_elements, margin, stride, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    input1 = tl.load(input1_ptr + offsets, mask=mask)
    input2 = tl.load(input2_ptr + offsets, mask=mask)
    target = tl.load(target_ptr + offsets, mask=mask)

    # Normalize input1 and input2
    norm1 = tl.sqrt(tl.sum(input1 * input1, axis=0))
    norm2 = tl.sqrt(tl.sum(input2 * input2, axis=0))
    input1_normalized = input1 / norm1
    input2_normalized = input2 / norm2

    # Compute cosine similarity
    cosine_similarity = tl.sum(input1_normalized * input2_normalized, axis=0)

    # Compute loss
    loss = tl.where(
        target == 1,
        1 - cosine_similarity,
        tl.maximum(0, cosine_similarity - margin)
    )

    # Store result
    tl.store(output_ptr + offsets, loss, mask=mask)

def fused_cosine_embedding_loss_with_normalization(
    input1: torch.Tensor, input2: torch.Tensor, target: torch.Tensor,
    margin: float = 0, reduction: str = 'mean'
) -> torch.Tensor:
    assert input1.shape == input2.shape, "Input tensors must have the same shape"
    assert target.shape[0] == input1.shape[0], "Target tensor must have the same batch size as inputs"

    n_elements = input1.shape[0] * input1.shape[1]
    output = torch.empty_like(target, dtype=input1.dtype)

    grid = (triton.cdiv(n_elements, 1024),)
    cosine_embedding_loss_kernel[grid](
        input1, input2, target, output,
        n_elements, margin, input1.stride(0),
        BLOCK_SIZE=1024
    )

    if reduction == 'mean':
        return output.mean()
    elif reduction == 'sum':
        return output.sum()
    elif reduction == 'none':
        return output
    else:
        raise ValueError(f"Invalid reduction type: {reduction}")

# Example usage:
# input1 = torch.randn(32, 128, device='cuda')
# input2 = torch.randn(32, 128, device='cuda')
# target = torch.randint(0, 2, (32,), device='cuda') * 2 - 1  # Random -1 or 1
# loss = fused_cosine_embedding_loss_with_normalization(input1, input2, target)
