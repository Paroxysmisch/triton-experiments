import torch
import triton
import triton.language as tl
import math

@triton.jit
def fused_embedding_kernel(
    out_ptr,  # pointer to the output
    indices_ptr,  # pointer to the input indices
    weight_ptr,  # pointer to the weights
    other_ptr,  # pointer to the other tensor
    padding_idx,  # padding index
    max_norm,  # max norm value
    norm_type,  # norm type for max norm
    scale_grad_by_freq,  # flag for scaling gradients by frequency
    sparse,  # flag for sparse gradients
    N: tl.constexpr,  # number of columns in X
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    out_ptr += pid * N
    indices_ptr += pid

    mask = tl.arange(0, BLOCK_SIZE) < N
    cols = tl.arange(0, BLOCK_SIZE)

    row_idx = tl.load(indices_ptr).to(tl.int32)
    weight_ptr += row_idx * N
    embedding_weight = tl.load(weight_ptr + cols, mask, other=0.0)

    # Apply max norm if specified
    if max_norm is not None:
        norm = tl.norm(embedding_weight, p=norm_type, dim=0)
        embedding_weight = tl.where(norm > max_norm, embedding_weight * (max_norm / norm), embedding_weight)

    # Load the other tensor and perform addition
    other_tensor = tl.load(other_ptr + pid * N, mask=mask, other=0.0)
    result = embedding_weight + other_tensor

    # Apply tanh activation
    output = tl.tanh(result)
    tl.store(out_ptr + cols, output, mask)

class FusedEmbeddingAddTanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_indices, weight, other, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False, out=None):
        M = math.prod(input_indices.shape)
        N = weight.shape[-1]

        BLOCK_SIZE = triton.next_power_of_2(N)
        input_indices = input_indices.contiguous()
        weight = weight.contiguous()
        other = other.contiguous()
        output = torch.empty((*input_indices.shape, N), device=input_indices.device, dtype=weight.dtype)

        with torch.cuda.device(weight.device):
            fused_embedding_kernel[M,](output, input_indices, weight, other, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse, N, BLOCK_SIZE)

        ctx.save_for_backward(input_indices, weight, other)
        ctx.padding_idx = padding_idx
        ctx.max_norm = max_norm
        ctx.norm_type = norm_type
        ctx.scale_grad_by_freq = scale_grad_by_freq
        ctx.sparse = sparse

        return output

    @staticmethod
    def backward(ctx, grad_outputs):
        input_indices, weight, other = ctx.saved_tensors
        grad_inputs = None  # Gradients for input_indices are not needed
        grad_weight = torch.zeros_like(weight)
        grad_other = grad_outputs.clone()  # Gradients for other tensor

        # Implement gradient computation for weight and other if needed
        # This part would require additional kernels for backward pass

        return grad_inputs, grad_weight, grad_other, None, None, None, None, None

def fused_embedding_add_tanh(input_indices, weight, other, *, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False, out=None):
    return FusedEmbeddingAddTanh.apply(input_indices, weight, other, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse, out)
