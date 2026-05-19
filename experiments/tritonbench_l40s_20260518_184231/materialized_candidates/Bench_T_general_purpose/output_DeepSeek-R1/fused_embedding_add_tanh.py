import torch
import torch.nn.functional as F
import triton
import triton.language as tl

class FusedEmbeddingAddTanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_indices, weight, other, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse):
        # Compute embedding with PyTorch to handle max_norm, padding_idx, etc.
        E = F.embedding(input_indices, weight, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse)
        S = E + other  # Broadcasting handled here
        Y = torch.tanh(S)
        ctx.save_for_backward(input_indices, weight, other, E, S, Y)
        ctx.padding_idx = padding_idx
        ctx.max_norm = max_norm
        ctx.norm_type = norm_type
        ctx.scale_grad_by_freq = scale_grad_by_freq
        ctx.sparse = sparse
        return Y

    @staticmethod
    def backward(ctx, grad_output):
        input_indices, weight, other, E, S, Y = ctx.saved_tensors
        padding_idx = ctx.padding_idx
        max_norm = ctx.max_norm
        norm_type = ctx.norm_type
        scale_grad_by_freq = ctx.scale_grad_by_freq
        sparse = ctx.sparse

        # Gradient for tanh: dY * (1 - Y^2)
        grad_S = grad_output * (1 - Y.pow(2))

        # Gradient for addition: grad_S is gradient for E and other
        grad_E = grad_S
        grad_other = grad_S  # Sum over broadcasted dimensions if needed

        # Compute gradient for embedding
        grad_weight = torch.zeros_like(weight)
        if ctx.needs_input_grad[1]:
            # Use PyTorch's embedding_backward to handle scale_grad_by_freq, sparse, etc.
            grad_weight = torch.embedding_backward(
                grad_E, input_indices, weight.size(0), padding_idx,
                scale_grad_by_freq, sparse
            )

        # Gradient for other (summing over broadcasted dimensions)
        if ctx.needs_input_grad[2]:
            # Sum gradients over dimensions that were broadcasted
            reduce_dims = []
            for dim in range(other.dim()):
                if other.shape[dim] == 1 and grad_S.shape[dim] != 1:
                    reduce_dims.append(dim)
            if reduce_dims:
                grad_other = grad_other.sum(dim=reduce_dims, keepdim=True)
            else:
                grad_other = grad_other

        return None, grad_weight, grad_other, None, None, None, None, None

def fused_embedding_add_tanh(input_indices, weight, other, *, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False, out=None):
    result = FusedEmbeddingAddTanh.apply(input_indices, weight, other, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse)
    if out is not None:
        out.copy_(result)
        return out
    return result
