import triton
import triton.language as tl

@triton.jit
def cosine_embedding_loss_kernel(
    input1_ptr, input2_ptr, target_ptr, output_ptr,
    stride_input1, stride_input2, stride_target, stride_output,
    n_elements, margin, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    input1 = tl.load(input1_ptr + offsets * stride_input1, mask=mask)
    input2 = tl.load(input2_ptr + offsets * stride_input2, mask=mask)
    target = tl.load(target_ptr + offsets * stride_target, mask=mask)

    # Compute dot product
    dot_product = tl.sum(input1 * input2, axis=0)

    # Compute cosine similarity
    cosine_similarity = dot_product

    # Compute loss
    loss = tl.where(target == 1, 1 - cosine_similarity, tl.max(0, margin - cosine_similarity))

    # Write output
    tl.store(output_ptr + offsets * stride_output, loss, mask=mask)

import torch
import triton
import triton.language as tl

def fused_cosine_embedding_loss_with_normalization(input1: torch.Tensor, input2: torch.Tensor, target: torch.Tensor, margin: float = 0, reduction: str = 'mean') -> torch.Tensor:
    # Ensure inputs are normalized along dimension 1
    input1 = torch.nn.functional.normalize(input1, p=2, dim=1)
    input2 = torch.nn.functional.normalize(input2, p=2, dim=1)

    # Ensure inputs and target are on the same device
    device = input1.device
    target = target.to(device)

    # Ensure inputs and target have the same number of elements
    assert input1.shape[0] == input2.shape[0] == target.shape[0], "Input and target must have the same number of elements"

    # Prepare output tensor
    output = torch.empty_like(target, device=device)

    # Define grid and block sizes
    n_elements = input1.shape[0]
    BLOCK_SIZE = 128
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    # Launch kernel
    cosine_embedding_loss_kernel[grid](
        input1, input2, target, output,
        input1.stride(0), input2.stride(0), target.stride(0), output.stride(0),
        n_elements, margin, BLOCK_SIZE
    )

    # Apply reduction
    if reduction == 'mean':
        return output.mean()
    elif reduction == 'sum':
        return output.sum()
    elif reduction == 'none':
        return output
    else:
        raise ValueError("Invalid reduction method. Choose from 'none', 'mean', or 'sum'.")

# Example usage
input1 = torch.randn(10, 5, device='cuda')
input2 = torch.randn(10, 5, device='cuda')
target = torch.randint(-1, 2, (10,), device='cuda') * 2 - 1  # Ensure target is either -1 or 1
loss = fused_cosine_embedding_loss_with_normalization(input1, input2, target, margin=0.5, reduction='mean')
print(loss)
