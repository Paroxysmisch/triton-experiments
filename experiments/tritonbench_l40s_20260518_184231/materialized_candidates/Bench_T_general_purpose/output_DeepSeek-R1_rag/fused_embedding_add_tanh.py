import torch
import triton
import triton.language as tl
import math

@triton.jit
def fused_embedding_add_tanh_forward_kernel(
    output_ptr,
    input_ptr,
    weight_ptr,
    other_ptr,
    M,
    N,
    other_stride_0,
    other_stride_1,
    max_norm,
    norm_type,
    HAS_MAX_NORM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= M:
        return

    # Load input index
    idx = tl.load(input_ptr + pid)
    
    # Initialize pointers for embedding row
    weight_row_ptr = weight_ptr + idx * N
    output_row_ptr = output_ptr + pid * N

    # Compute embedding and apply max_norm if needed
    if HAS_MAX_NORM:
        # Compute norm of the embedding row
        norm = 0.0
        for col_off in range(0, N, BLOCK_SIZE):
            cols = col_off + tl.arange(0, BLOCK_SIZE)
            mask = cols < N
            w = tl.load(weight_row_ptr + cols, mask=mask, other=0.0)
            norm += tl.sum(tl.pow(tl.abs(w), norm_type) * mask)
        norm = tl.pow(norm, 1.0 / norm_type)
        scale = tl.minimum(1.0, max_norm / (norm + 1e-8))
    else:
        scale = 1.0

    # Process each block of the embedding row
    for col_off in range(0, N, BLOCK_SIZE):
        cols = col_off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        # Load embedding weight
        w = tl.load(weight_row_ptr + cols, mask=mask, other=0.0)
        if HAS_MAX_NORM:
            w = w * scale
        # Load other element with broadcasting
        other_offset = pid * other_stride_0 + cols * other_stride_1
        o = tl.load(other_ptr + other_offset, mask=mask, other=0.0)
        # Compute and store result
        s = w + o
        y = tl.math.tanh(s)
        tl.store(output_row_ptr + cols, y, mask=mask)

@triton.jit
def fused_embedding_add_tanh_backward_kernel(
    grad_weight_ptr,
    grad_other_ptr,
    grad_output_ptr,
    input_ptr,
    other_stride_0,
    other_stride_1,
    M,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= M:
        return

    idx = tl.load(input_ptr + pid)
    grad_output_row = grad_output_ptr + pid * N

    # Compute gradient for embedding
    for col_off in range(0, N, BLOCK_SIZE):
        cols = col_off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        # Load gradient of output
        grad_y = tl.load(grad_output_row + cols, mask=mask, other=0.0)
        # Gradient of tanh: grad_s = grad_y * (1 - y^2)
        y = tl.load(grad_output_row + cols, mask=mask, other=0.0)
        grad_s = grad_y * (1 - y * y)
        # Accumulate gradient for weight
        grad_w = grad_s
        grad_weight_idx = grad_weight_ptr + idx * N + cols
        tl.atomic_add(grad_weight_idx, grad_w, mask=mask)
        # Accumulate gradient for other
        other_offset = pid * other_stride_0 + cols * other_stride_1
        grad_o = grad_s
        tl.atomic_add(grad_other_ptr + other_offset, grad_o, mask=mask)

class FusedEmbeddingAddTanhFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_indices, weight, other, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse):
        # Flatten input indices
        input_shape = input_indices.shape
        input_flat = input_indices.view(-1)
        M = input_flat.size(0)
        N = weight.size(1)
        
        # Broadcast other to (M, N)
        other_expanded = other.broadcast_to((M, N))
        other_stride_0 = other_expanded.stride(0)
        other_stride_1 = other_expanded.stride(1)
        
        # Output tensor
        output = torch.empty((M, N), device=input_indices.device, dtype=weight.dtype)
        
        # Kernel configuration
        BLOCK_SIZE = triton.next_power_of_2(N)
        grid = (M,)
        
        # Compute max_norm
        has_max_norm = max_norm is not None
        max_norm_val = max_norm if has_max_norm else 0.0
        
        # Launch kernel
        fused_embedding_add_tanh_forward_kernel[grid](
            output, input_flat, weight, other_expanded,
            M, N, other_stride_0, other_stride_1,
            max_norm_val, norm_type, has_max_norm,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        # Save for backward
        ctx.save_for_backward(input_flat, weight, other_expanded)
        ctx.input_shape = input_shape
        ctx.padding_idx = padding_idx
        ctx.scale_grad_by_freq = scale_grad_by_freq
        ctx.sparse = sparse
        ctx.max_norm = max_norm
        ctx.norm_type = norm_type
        ctx.other_stride_0 = other_stride_0
        ctx.other_stride_1 = other_stride_1
        
        return output.view(*input_shape, N)

    @staticmethod
    def backward(ctx, grad_output):
        input_flat, weight, other_expanded = ctx.saved_tensors
        M, N = input_flat.size(0), weight.size(1)
        grad_output_flat = grad_output.contiguous().view(-1, N)
        
        # Initialize gradients
        grad_weight = torch.zeros_like(weight)
        grad_other = torch.zeros_like(other_expanded)
        
        BLOCK_SIZE = triton.next_power_of_2(N)
        grid = (M,)
        
        fused_embedding_add_tanh_backward_kernel[grid](
            grad_weight, grad_other, grad_output_flat, input_flat,
            ctx.other_stride_0, ctx.other_stride_1,
            M, N, BLOCK_SIZE=BLOCK_SIZE
        )
        
        # Handle padding_idx
        if ctx.padding_idx is not None:
            grad_weight[ctx.padding_idx] = 0.0
        
        # Scale gradients by frequency if needed
        if ctx.scale_grad_by_freq:
            # Compute frequency (simplified example)
            unique_indices, counts = torch.unique(input_flat, return_counts=True)
            grad_weight[unique_indices] /= counts.float().unsqueeze(1)
        
        # Sum grad_other according to broadcasted dimensions
        grad_other = grad_other.sum(dim=0, keepdim=True) if other_expanded.dim() < grad_output.dim() else grad_other
        
        return grad_output, grad_weight, grad_other, None, None, None, None, None

def fused_embedding_add_tanh(input_indices, weight, other, *, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False, out=None):
    result = FusedEmbeddingAddTanhFunction.apply(
        input_indices, weight, other,
        padding_idx, max_norm, norm_type,
        scale_grad_by_freq, sparse
    )
    if out is not None:
        out.copy_(result)
        return out
    return result
