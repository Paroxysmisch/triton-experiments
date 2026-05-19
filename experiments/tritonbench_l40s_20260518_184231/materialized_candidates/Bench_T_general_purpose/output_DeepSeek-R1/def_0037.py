import torch
import triton
import triton.language as tl

@triton.jit
def cosine_embedding_loss_kernel(
    input1_ptr, input2_ptr, target_ptr, output_ptr,
    margin, n_rows, row_size,
    input1_row_stride, input2_row_stride, target_stride, output_stride,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    if row_idx >= n_rows:
        return
    
    # Compute L2 norm for input1
    input1_norm = 0.0
    for col_offset in range(0, row_size, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < row_size
        a = tl.load(input1_ptr + row_idx * input1_row_stride + col_idx, mask=mask, other=0.0)
        input1_norm += tl.sum(a * a)
    input1_norm = tl.sqrt(input1_norm + 1e-8)
    input1_inv_norm = 1.0 / input1_norm
    
    # Compute L2 norm for input2
    input2_norm = 0.0
    for col_offset in range(0, row_size, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < row_size
        b = tl.load(input2_ptr + row_idx * input2_row_stride + col_idx, mask=mask, other=0.0)
        input2_norm += tl.sum(b * b)
    input2_norm = tl.sqrt(input2_norm + 1e-8)
    input2_inv_norm = 1.0 / input2_norm
    
    # Compute dot product of normalized inputs
    dot_product = 0.0
    for col_offset in range(0, row_size, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < row_size
        a = tl.load(input1_ptr + row_idx * input1_row_stride + col_idx, mask=mask, other=0.0) * input1_inv_norm
        b = tl.load(input2_ptr + row_idx * input2_row_stride + col_idx, mask=mask, other=0.0) * input2_inv_norm
        dot_product += tl.sum(a * b)
    
    # Load target value
    target_val = tl.load(target_ptr + row_idx * target_stride)
    
    # Compute loss
    if target_val == 1.0:
        loss = 1.0 - dot_product
    else:
        loss = tl.maximum(dot_product - margin, 0.0)
    
    # Store loss
    tl.store(output_ptr + row_idx * output_stride, loss)

def fused_cosine_embedding_loss_with_normalization(
    input1: torch.Tensor,
    input2: torch.Tensor,
    target: torch.Tensor,
    margin: float = 0,
    reduction: str = 'mean'
) -> torch.Tensor:
    assert input1.dim() == 2, "input1 must be 2D"
    assert input2.dim() == 2, "input2 must be 2D"
    assert input1.size(0) == input2.size(0), "input1 and input2 must have the same number of rows"
    assert input1.size(1) == input2.size(1), "input1 and input2 must have the same feature dimension"
    assert target.dim() == 1, "target must be 1D"
    assert target.size(0) == input1.size(0), "target must have the same number of elements as input1 has rows"
    assert reduction in ['none', 'mean', 'sum'], "reduction must be 'none', 'mean', or 'sum'"
    
    n_rows, row_size = input1.size(0), input1.size(1)
    
    # Ensure contiguous tensors
    input1 = input1.contiguous()
    input2 = input2.contiguous()
    target = target.contiguous()
    
    # Allocate output tensor
    output = torch.empty(n_rows, device=input1.device, dtype=input1.dtype)
    
    # Launch kernel
    BLOCK_SIZE = 1024  # can be tuned for optimal performance
    grid = (n_rows,)
    cosine_embedding_loss_kernel[grid](
        input1, input2, target, output,
        margin,
        n_rows, row_size,
        input1.stride(0), input2.stride(0), target.stride(0), output.stride(0),
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Apply reduction
    if reduction == 'none':
        return output
    elif reduction == 'mean':
        return output.mean()
    elif reduction == 'sum':
        return output.sum()
