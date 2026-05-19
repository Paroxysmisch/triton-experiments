import torch
import triton
import triton.language as tl
import math

@triton.jit
def fused_cosine_embedding_loss_kernel(
    # Pointers to input/output tensors
    input1_ptr, input2_ptr, target_ptr, output_ptr,
    # Tensor dimensions and properties
    batch_size, feat_size, margin,
    input1_stride_b, input1_stride_f,
    input2_stride_b, input2_stride_f,
    target_stride,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Compute batch index
    batch_idx = pid
    
    # Load input vectors for this batch
    input1_off = batch_idx * input1_stride_b
    input2_off = batch_idx * input2_stride_b
    
    # Initialize accumulators for dot product and norms
    dot_prod = 0.0
    norm1 = 0.0
    norm2 = 0.0
    
    # Compute dot product and norms
    for idx in range(0, feat_size, BLOCK_SIZE):
        block_size = min(BLOCK_SIZE, feat_size - idx)
        
        # Load blocks of vectors
        x1 = tl.load(input1_ptr + input1_off + idx * input1_stride_f, mask=idx < feat_size, other=0.0)
        x2 = tl.load(input2_ptr + input2_off + idx * input2_stride_f, mask=idx < feat_size, other=0.0)
        
        # Update dot product and norms
        dot_prod += tl.sum(x1 * x2, axis=0)
        norm1 += tl.sum(x1 * x1, axis=0)
        norm2 += tl.sum(x2 * x2, axis=0)
    
    # Normalize dot product
    norm1 = tl.sqrt(norm1)
    norm2 = tl.sqrt(norm2)
    cos_sim = dot_prod / (norm1 * norm2 + 1e-8)
    
    # Load target value
    target = tl.load(target_ptr + batch_idx * target_stride)
    
    # Compute loss
    loss = tl.where(
        target > 0,
        1.0 - cos_sim,
        tl.maximum(0.0, cos_sim - margin)
    )
    
    # Store result
    tl.store(output_ptr + batch_idx, loss)

def fused_cosine_embedding_loss_with_normalization(
    input1: torch.Tensor,
    input2: torch.Tensor,
    target: torch.Tensor,
    margin: float = 0.0,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Computes cosine embedding loss between two normalized tensors.
    
    Args:
        input1 (Tensor): First input tensor to be normalized and compared
        input2 (Tensor): Second input tensor to be normalized and compared
        target (Tensor): Tensor with values 1 or -1
        margin (float, optional): Margin for dissimilarity. Default: 0
        reduction (str, optional): Reduction method ('none'|'mean'|'sum'). Default: 'mean'
    
    Returns:
        Tensor: Computed loss
    """
    assert input1.dim() == 2 and input2.dim() == 2, "Input tensors must be 2D"
    assert input1.size() == input2.size(), "Input tensors must have same size"
    assert target.dim() == 1 and target.size(0) == input1.size(0), "Target size mismatch"
    assert reduction in ['none', 'mean', 'sum'], "Invalid reduction method"
    
    batch_size, feat_size = input1.shape
    
    # Ensure inputs are contiguous
    input1 = input1.contiguous()
    input2 = input2.contiguous()
    target = target.contiguous()
    
    # Initialize output tensor
    output = torch.empty(batch_size, device=input1.device, dtype=input1.dtype)
    
    # Launch kernel
    grid = (batch_size,)
    fused_cosine_embedding_loss_kernel[grid](
        input1_ptr=input1,
        input2_ptr=input2,
        target_ptr=target,
        output_ptr=output,
        batch_size=batch_size,
        feat_size=feat_size,
        margin=margin,
        input1_stride_b=input1.stride(0),
        input1_stride_f=input1.stride(1),
        input2_stride_b=input2.stride(0),
        input2_stride_f=input2.stride(1),
        target_stride=target.stride(0),
        BLOCK_SIZE=128,
    )
    
    # Apply reduction
    if reduction == 'none':
        return output
    elif reduction == 'mean':
        return output.mean()
    else:  # sum
        return output.sum()
