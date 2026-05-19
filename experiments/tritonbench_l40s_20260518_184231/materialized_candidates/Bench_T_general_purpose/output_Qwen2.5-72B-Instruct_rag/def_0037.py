import triton
import triton.language as tl

@triton.jit
def l2_normalize_kernel(input_ptr, output_ptr, n_elements, stride, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input_vec = tl.load(input_ptr + offsets * stride, mask=mask)
    
    # Compute L2 norm
    norm = tl.sqrt(tl.sum(input_vec * input_vec, axis=0))
    norm = tl.where(norm == 0, 1, norm)  # Avoid division by zero
    output_vec = input_vec / norm
    
    tl.store(output_ptr + offsets * stride, output_vec, mask=mask)

@triton.jit
def cosine_embedding_loss_kernel(input1_ptr, input2_ptr, target_ptr, output_ptr, n_elements, margin, reduction, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    input1_vec = tl.load(input1_ptr + offsets, mask=mask)
    input2_vec = tl.load(input2_ptr + offsets, mask=mask)
    target_vec = tl.load(target_ptr + offsets, mask=mask)
    
    # Compute cosine similarity
    dot_product = tl.sum(input1_vec * input2_vec, axis=0)
    norm1 = tl.sqrt(tl.sum(input1_vec * input1_vec, axis=0))
    norm2 = tl.sqrt(tl.sum(input2_vec * input2_vec, axis=0))
    norm1 = tl.where(norm1 == 0, 1, norm1)  # Avoid division by zero
    norm2 = tl.where(norm2 == 0, 1, norm2)  # Avoid division by zero
    cosine_similarity = dot_product / (norm1 * norm2)
    
    # Compute loss
    loss = 1 - cosine_similarity
    loss = tl.where(target_vec == 1, loss, tl.max(0, margin - cosine_similarity))
    
    if reduction == 'mean':
        loss = tl.sum(loss) / n_elements
    elif reduction == 'sum':
        loss = tl.sum(loss)
    
    tl.store(output_ptr + offsets, loss, mask=mask)

import torch

def fused_cosine_embedding_loss_with_normalization(input1: torch.Tensor, input2: torch.Tensor, target: torch.Tensor, margin: float = 0, reduction: str = 'mean') -> torch.Tensor:
    assert input1.shape == input2.shape, "Input tensors must have the same shape"
    assert target.shape == input1.shape[:-1], "Target tensor must have the same shape as the first dimension of input tensors"
    assert reduction in ['none', 'mean', 'sum'], "Reduction must be 'none', 'mean', or 'sum'"
    
    # Normalize input tensors along dimension 1
    input1_normalized = torch.empty_like(input1)
    input2_normalized = torch.empty_like(input2)
    
    grid = lambda meta: (input1.numel() // meta['BLOCK_SIZE'],)
    l2_normalize_kernel[grid](input1, input1_normalized, input1.numel(), input1.stride(1), BLOCK_SIZE=1024)
    l2_normalize_kernel[grid](input2, input2_normalized, input2.numel(), input2.stride(1), BLOCK_SIZE=1024)
    
    # Compute cosine embedding loss
    output = torch.empty(input1.shape[:-1], device=input1.device)
    cosine_embedding_loss_kernel[grid](input1_normalized, input2_normalized, target, output, input1.numel(), margin, reduction, BLOCK_SIZE=1024)
    
    if reduction == 'none':
        return output
    else:
        return output.mean() if reduction == 'mean' else output.sum()
