import torch
import torch.nn.functional as F
import triton
import triton.language as tl

class FusedGatherMaskedFill(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, dim, index, mask, value, sparse_grad=False):
        ctx.save_for_backward(index, mask)
        ctx.dim = dim
        ctx.sparse_grad = sparse_grad
        ctx.input_shape = input.shape
        
        # Compute the gathered tensor
        gathered = torch.gather(input, dim, index)
        # Apply the masked fill
        output = gathered.masked_fill(mask, value)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        index, mask = ctx.saved_tensors
        dim = ctx.dim
        sparse_grad = ctx.sparse_grad
        
        # Zero out gradients where the mask was applied
        grad_output = grad_output.clone()
        grad_output[mask] = 0
        
        # Scatter the gradients back to the input's shape
        grad_input = torch.zeros(ctx.input_shape, dtype=grad_output.dtype, device=grad_output.device)
        grad_input.scatter_add_(dim, index, grad_output)
        
        if sparse_grad:
            grad_input = grad_input.to_sparse()
        
        return grad_input, None, None, None, None, None, None

def fused_gather_masked_fill(input, dim, index, mask, value, *, sparse_grad=False, out=None):
    # Check input and index have the same number of dimensions
    if input.dim() != index.dim():
        raise ValueError("input and index must have the same number of dimensions")
    # Check index dimensions (except dim) are <= input's
    for d in range(input.dim()):
        if d != dim and index.size(d) > input.size(d):
            raise ValueError(f"Index size at dim {d} exceeds input size")
    # Broadcast mask to the gathered output shape
    output_shape = index.shape
    try:
        mask = mask.broadcast_to(output_shape)
    except RuntimeError:
        raise ValueError("mask is not broadcastable to the gathered output shape")
    # Ensure contiguous tensors
    input = input.contiguous()
    index = index.contiguous()
    mask = mask.contiguous()
    # Apply the fused operation using the custom autograd Function
    result = FusedGatherMaskedFill.apply(input, dim, index, mask, value, sparse_grad)
    # Handle the output tensor if provided
    if out is not None:
        out.copy_(result)
        return out
    return result
