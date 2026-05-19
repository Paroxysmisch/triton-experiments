import torch
import triton
import triton.language as tl

@triton.jit
def fused_embedding_add_tanh_kernel(
    out_ptr,  # pointer to the output
    indices_ptr,  # pointer to the input indices
    weight_ptr,  # pointer to the embedding weights
    other_ptr,  # pointer to the tensor to add
    V: tl.constexpr,  # number of embeddings (vocabulary size)
    D: tl.constexpr,  # embedding dimension
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    row_idx = tl.load(indices_ptr + pid)

    # Check if the row index is within bounds
    if row_idx < V:
        weight_offset = row_idx * D
        other_offset = pid * D

        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < D

        # Load embedding and other tensor, then add them
        embedding = tl.load(weight_ptr + weight_offset + cols, mask=mask, other=0.0)
        other = tl.load(other_ptr + other_offset + cols, mask=mask, other=0.0)
        summed = embedding + other

        # Apply tanh activation
        activated = tl.tanh(summed)

        # Store the result
        tl.store(out_ptr + other_offset + cols, activated, mask=mask)

class FusedEmbeddingAddTanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_indices, weight, other, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False, out=None):
        V, D = weight.shape
        M = input_indices.numel()

        # Handle max_norm if specified
        if max_norm is not None:
            with torch.no_grad():
                norms = torch.norm(weight, p=norm_type, dim=1, keepdim=True)
                mask = norms > max_norm
                weight[mask] = weight[mask] * (max_norm / norms[mask])

        # Prepare output tensor
        if out is None:
            out = torch.empty((M, D), device=weight.device, dtype=weight.dtype)

        # Launch Triton kernel
        BLOCK_SIZE = triton.next_power_of_2(D)
        grid = lambda meta: (M,)
        fused_embedding_add_tanh_kernel[grid](
            out,
            input_indices,
            weight,
            other,
            V,
            D,
            BLOCK_SIZE,
        )

        ctx.save_for_backward(input_indices, weight, other)
        ctx.padding_idx = padding_idx
        ctx.scale_grad_by_freq = scale_grad_by_freq
        ctx.sparse = sparse

        return out

    @staticmethod
    def backward(ctx, grad_output):
        input_indices, weight, other = ctx.saved_tensors
        # Implement the backward pass here if needed
        # For simplicity, this example does not implement backward logic
        return None, None, None, None, None, None, None, None, None

def fused_embedding_add_tanh(input_indices, weight, other, *, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False, out=None):
    return FusedEmbeddingAddTanh.apply(input_indices, weight, other, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse, out)
