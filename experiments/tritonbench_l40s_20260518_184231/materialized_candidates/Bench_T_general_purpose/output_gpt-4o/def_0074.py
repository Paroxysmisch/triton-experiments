import triton
import triton.language as tl
import torch

@triton.jit
def cosine_similarity_kernel(
    x1_ptr, x2_ptr, out_ptr, 
    dim, eps_similarity, eps_norm, p_norm, 
    stride_x1, stride_x2, stride_out, 
    n_elements
):
    # Get the program ID for parallel execution
    pid = tl.program_id(0)
    
    # Calculate the index for this program
    idx = pid * tl.num_programs(0) + tl.arange(0, n_elements)
    
    # Load elements from x1 and x2
    x1 = tl.load(x1_ptr + idx * stride_x1, mask=idx < n_elements)
    x2 = tl.load(x2_ptr + idx * stride_x2, mask=idx < n_elements)
    
    # Compute L_p norm for x1 and x2
    norm_x1 = tl.pow(tl.abs(x1), p_norm).sum()**(1/p_norm)
    norm_x2 = tl.pow(tl.abs(x2), p_norm).sum()**(1/p_norm)
    
    # Normalize x1 and x2
    norm_x1 = tl.max(norm_x1, eps_norm)
    norm_x2 = tl.max(norm_x2, eps_norm)
    
    x1_normalized = x1 / norm_x1
    x2_normalized = x2 / norm_x2
    
    # Compute dot product
    dot_product = (x1_normalized * x2_normalized).sum()
    
    # Compute cosine similarity
    similarity = dot_product / tl.max(norm_x1 * norm_x2, eps_similarity)
    
    # Store the result
    tl.store(out_ptr + pid * stride_out, similarity)

def normalized_cosine_similarity(x1: torch.Tensor, x2: torch.Tensor, dim: int = 1, eps_similarity: float = 1e-8, p_norm: float = 2, eps_norm: float = 1e-12) -> torch.Tensor:
    # Ensure x2 can be broadcasted to x1
    if x2.shape != x1.shape:
        x2 = x2.expand_as(x1)
    
    # Prepare output tensor
    output_shape = list(x1.shape)
    output_shape[dim] = 1
    output = torch.empty(output_shape, device=x1.device, dtype=x1.dtype)
    
    # Calculate strides
    stride_x1 = x1.stride(dim)
    stride_x2 = x2.stride(dim)
    stride_out = output.stride(dim)
    
    # Launch Triton kernel
    n_elements = x1.size(dim)
    grid = (output.numel(),)
    
    cosine_similarity_kernel[grid](
        x1, x2, output,
        dim, eps_similarity, eps_norm, p_norm,
        stride_x1, stride_x2, stride_out,
        n_elements
    )
    
    return output.squeeze(dim)

# Example usage
x1 = torch.rand((4, 5), device='cuda')
x2 = torch.rand((4, 5), device='cuda')
similarity = normalized_cosine_similarity(x1, x2)
print(similarity)
