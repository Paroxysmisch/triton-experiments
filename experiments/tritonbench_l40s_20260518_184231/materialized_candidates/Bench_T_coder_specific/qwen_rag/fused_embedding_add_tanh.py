import triton
import triton.language as tl

# Define constants
BLOCK_SIZE = 256

@triton.jit
def fused_embedding_add_tanh_kernel(
    out_ptr,  # Pointer to the output
    in_ptr,   # Pointer to the input indices
    weight_ptr,  # Pointer to the embedding weight matrix
    other_ptr,  # Pointer to the tensor to be added
    padding_idx,  # Padding index
    M: tl.constexpr,  # Number of elements in the input indices
    V: tl.constexpr,  # Vocabulary size
    D: tl.constexpr,  # Embedding dimension
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_size = tl.cdiv(M, BLOCK_SIZE)
    offset = pid * BLOCK_SIZE

    # Load input indices
    indices = tl.load(in_ptr + offset, mask=offset < M, other=0)
    
    # Calculate embedding indices
    emb_indices = indices * V
    
    # Load embeddings
    embedding_weights = tl.load(weight_ptr + emb_indices, mask=offset < M, other=0.0)
    
    # Load other tensor
    other_values = tl.load(other_ptr + offset, mask=offset < M, other=0.0)
    
    # Perform element-wise addition
    sum_values = embedding_weights + other_values
    
    # Apply tanh activation
    tanh_values = tl.math.tanh(sum_values)
    
    # Store results
    tl.store(out_ptr + offset, tanh_values, mask=offset < M)

@triton.jit
def fused_embedding_add_tanh_forward(
    input_indices,  # Input indices tensor
    weight,         # Embedding weight matrix
    other,          # Tensor to be added
    padding_idx,    # Optional padding index
    max_norm,       # Optional max norm for embeddings
    norm_type,      # Norm type for max norm
    scale_grad_by_freq,  # Scale gradients by frequency
    sparse,         # Sparse gradients
    out             # Output tensor
):
    M = input_indices.numel()
    V = weight.size(0)
    D = weight.size(1)
    
    fused_embedding_add_tanh_kernel[tl.cdiv(M, BLOCK_SIZE), BLOCK_SIZE](
        out.data_ptr(),
        input_indices.data_ptr(),
        weight.data_ptr(),
        other.data_ptr(),
        padding_idx,
        M,
        V,
        D,
        BLOCK_SIZE
    )

    return out

def fused_embedding_add_tanh(
    input_indices,  # LongTensor: Tensor containing indices into the embedding matrix, of arbitrary shape (*)
    weight,         # Tensor: The embedding matrix of shape (V, D)
    other,          # Tensor: Tensor to be added to the embeddings, must be broadcastable to the shape of E
    *,
    padding_idx=None,  # int, optional: If specified, the entries at `padding_idx` do not contribute to the gradient
    max_norm=None,     # float, optional: If given, each embedding vector with norm larger than `max_norm` is renormalized to have norm `max_norm`
    norm_type=2.0,     # float, optional: The p-norm to compute for the `max_norm` option. Default: `2.0`
    scale_grad_by_freq=False,  # bool, optional: If `True`, scale gradients by the inverse of frequency of the words in the mini-batch. Default: `False`
    sparse=False,      # bool, optional: If `True`, gradient w.r.t. `weight` will be a sparse tensor. Default: `False`
    out=None           # Tensor, optional: Output tensor. Ignored if `None`. Default: `None`
):
    if out is None:
        out = torch.empty_like(other)
    
    fused_embedding_add_tanh_forward(
        input_indices,
        weight,
        other,
        padding_idx,
        max_norm,
        norm_type,
        scale_grad_by_freq,
        sparse,
        out
    )
    
    return out
