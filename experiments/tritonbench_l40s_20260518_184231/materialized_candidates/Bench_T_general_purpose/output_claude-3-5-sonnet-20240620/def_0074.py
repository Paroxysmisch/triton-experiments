import triton
import triton.language as tl

@triton.jit
def cosine_similarity_kernel(x1_ptr, x2_ptr, output_ptr, n_elements, dim, eps_norm, eps_similarity):
    # Calculate the normalization for x1
    x1 = tl.load(x1_ptr)
    norm_x1 = tl.norm(x1, p=2, dim=dim, keepdim=True)
    norm_x1 = tl.where(norm_x1 < eps_norm, eps_norm, norm_x1)
    normalized_x1 = x1 / norm_x1

    # Calculate the normalization for x2
    x2 = tl.load(x2_ptr)
    norm_x2 = tl.norm(x2, p=2, dim=dim, keepdim=True)
    norm_x2 = tl.where(norm_x2 < eps_norm, eps_norm, norm_x2)
    normalized_x2 = x2 / norm_x2

    # Compute the cosine similarity
    similarity = tl.dot(normalized_x1, normalized_x2)
    max_norm_x1 = tl.max(norm_x1, eps_similarity)
    max_norm_x2 = tl.max(norm_x2, eps_similarity)
    output = similarity / (max_norm_x1 * max_norm_x2)

    # Store the result
    tl.store(output_ptr, output)

def normalized_cosine_similarity(x1: Tensor, x2: Tensor, dim: int = 1, eps_similarity: float = 1e-8, p_norm: float = 2, eps_norm: float = 1e-12) -> Tensor:
    # Ensure x2 is broadcastable to x1's shape
    x1_shape = x1.shape
    x2_shape = x2.shape
    if x1_shape[dim] != x2_shape[dim]:
        x2 = x2.expand(x1_shape)

    # Allocate output tensor
    output = torch.empty(x1_shape, dtype=x1.dtype, device=x1.device)

    # Launch the Triton kernel
    grid = (x1_shape[dim],)  # Adjust grid size based on the dimension
    cosine_similarity_kernel[grid](x1, x2, output, x1_shape[dim], dim, eps_norm, eps_similarity)

    return output
