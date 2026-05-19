import triton
import triton.language as tl

@triton.jit
def normalize_kernel(x, y, dim, eps):
    """
    Normalizes the tensor x along the specified dimension.
    
    Args:
        x (float32[]): Input tensor to be normalized.
        y (float32[]): Output tensor where the normalized values will be stored.
        dim (int): Dimension along which to normalize.
        eps (float): Small value to avoid division by zero.
    """
    pid = tl.program_id(0)
    grid_size = tl.cdiv(x.shape[0], tl.block_dim(0))
    idx = pid * tl.block_dim(0) + tl.arange(0, tl.block_dim(0))

    # Normalize each element along the specified dimension
    norm = tl.sum(tl.square(x), axis=dim, keepdims=True)
    norm = tl.maximum(norm, eps)
    y[idx] = x[idx] / norm[idx]

@triton.jit
def cosine_similarity_kernel(x1, x2, y, dim, eps_similarity, eps_norm):
    """
    Computes the cosine similarity between two normalized tensors x1 and x2 along the specified dimension.
    
    Args:
        x1 (float32[]): First input tensor.
        x2 (float32[]): Second input tensor.
        y (float32[]): Output tensor where the cosine similarity values will be stored.
        dim (int): Dimension along which to compute the cosine similarity.
        eps_similarity (float): Small value to avoid division by zero in similarity computation.
        eps_norm (float): Small value to avoid division by zero in normalization.
    """
    pid = tl.program_id(0)
    grid_size = tl.cdiv(x1.shape[0], tl.block_dim(0))
    idx = pid * tl.block_dim(0) + tl.arange(0, tl.block_dim(0))

    # Normalize x1 and x2
    x1_normalized = tl.zeros_like(x1)
    x2_normalized = tl.zeros_like(x2)
    normalize_kernel[x1_normalized.numel()](x1, x1_normalized, dim, eps_norm)
    normalize_kernel[x2_normalized.numel()](x2, x2_normalized, dim, eps_norm)

    # Compute cosine similarity
    dot_product = tl.sum(x1_normalized * x2_normalized, axis=dim)
    norm_x1 = tl.sqrt(tl.sum(tl.square(x1_normalized), axis=dim))
    norm_x2 = tl.sqrt(tl.sum(tl.square(x2_normalized), axis=dim))
    similarity = dot_product / (tl.maximum(norm_x1, eps_similarity) * tl.maximum(norm_x2, eps_similarity))

    # Store the result
    y[idx] = similarity[idx]

# Wrapper function
def normalized_cosine_similarity(x1, x2, dim=1, eps_similarity=1e-8, p_norm=2, eps_norm=1e-12):
    """
    Computes the cosine similarity between two normalized input tensors x1 and x2.
    
    Args:
        x1 (Tensor): First input tensor.
        x2 (Tensor): Second input tensor.
        dim (int): Dimension along which to compute the cosine similarity. Default is 1.
        eps_similarity (float): Small value to avoid division by zero in similarity computation. Default is 1e-8.
        p_norm (float): P-norm used for normalization. Default is 2.
        eps_norm (float): Small value to avoid division by zero in normalization. Default is 1e-12.
    
    Returns:
        Tensor: Cosine similarity between the two normalized tensors.
    """
    output_shape = list(x1.shape)
    output_shape[dim] = 1
    y = tl.tensor(shape=output_shape, dtype=x1.dtype)
    
    cosine_similarity_kernel[x1.shape[0]](x1, x2, y, dim, eps_similarity, eps_norm)
    
    return y

# Example usage
if __name__ == "__main__":
    import torch
    
    x1 = torch.randn(10, 5).to('cuda')
    x2 = torch.randn(10, 5).to('cuda')
    
    result = normalized_cosine_similarity(x1, x2)
    print(result)
