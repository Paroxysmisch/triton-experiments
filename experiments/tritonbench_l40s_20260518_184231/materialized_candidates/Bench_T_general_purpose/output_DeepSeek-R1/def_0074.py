import torch
import triton
import triton.language as tl

@triton.jit
def normalized_cosine_similarity_kernel(
    x1_ptr, x2_ptr, dot_out_ptr, x1_norm_out_ptr, x2_norm_out_ptr,
    reduction_dim,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    dot_acc = tl.zeros((1,), dtype=tl.float32)
    x1_sq_acc = tl.zeros((1,), dtype=tl.float32)
    x2_sq_acc = tl.zeros((1,), dtype=tl.float32)
    
    for i in range(0, reduction_dim, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < reduction_dim
        
        x1 = tl.load(x1_ptr + pid * reduction_dim + offsets, mask=mask, other=0.0)
        x2 = tl.load(x2_ptr + pid * reduction_dim + offsets, mask=mask, other=0.0)
        
        dot_acc += tl.sum(x1 * x2)
        x1_sq_acc += tl.sum(x1 * x1)
        x2_sq_acc += tl.sum(x2 * x2)
    
    tl.store(dot_out_ptr + pid, dot_acc.to(x1.dtype))
    tl.store(x1_norm_out_ptr + pid, x1_sq_acc.to(x1.dtype))
    tl.store(x2_norm_out_ptr + pid, x2_sq_acc.to(x1.dtype))

def normalized_cosine_similarity(x1: torch.Tensor, x2: torch.Tensor, dim: int = 1, eps_similarity: float = 1e-8, p_norm: float = 2, eps_norm: float = 1e-12) -> torch.Tensor:
    # Broadcast x2 to match x1's shape
    x2 = x2.broadcast_to(x1.shape)
    
    # Normalize x1 and x2 using L_p norm along the specified dimension
    norm1 = x1.norm(p=p_norm, dim=dim, keepdim=True).clamp_min(eps_norm)
    norm2 = x2.norm(p=p_norm, dim=dim, keepdim=True).clamp_min(eps_norm)
    x1_normalized = x1 / norm1
    x2_normalized = x2 / norm2
    
    # Permute the reduction dimension to the end and flatten other dimensions
    original_shape = x1_normalized.shape
    perm = list(range(x1_normalized.dim()))
    perm.pop(dim)
    perm.append(dim)
    x1_perm = x1_normalized.permute(perm)
    x2_perm = x2_normalized.permute(perm)
    x1_flat = x1_perm.reshape(-1, x1_perm.size(-1))
    x2_flat = x2_perm.reshape(-1, x2_perm.size(-1))
    reduction_dim_size = x1_flat.size(1)
    
    # Allocate output tensors
    device = x1.device
    dtype = x1.dtype
    dot_output = torch.empty(x1_flat.size(0), dtype=dtype, device=device)
    x1_norm_output = torch.empty_like(dot_output)
    x2_norm_output = torch.empty_like(dot_output)
    
    # Launch kernel
    BLOCK_SIZE = 1024  # Tunable for optimal performance
    grid = (x1_flat.size(0),)
    normalized_cosine_similarity_kernel[grid](
        x1_flat, x2_flat, dot_output, x1_norm_output, x2_norm_output,
        reduction_dim_size, BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Compute similarity
    x1_norm = torch.sqrt(x1_norm_output).clamp_min(eps_similarity)
    x2_norm = torch.sqrt(x2_norm_output).clamp_min(eps_similarity)
    similarity = dot_output / (x1_norm * x2_norm)
    
    # Reshape back to original shape (excluding reduced dimension)
    result_shape = list(original_shape)
    result_shape.pop(dim)
    return similarity.reshape(result_shape)
