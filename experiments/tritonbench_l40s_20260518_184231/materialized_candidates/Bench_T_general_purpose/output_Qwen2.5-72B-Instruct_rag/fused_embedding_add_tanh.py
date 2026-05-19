import torch
import triton
import triton.language as tl

@triton.jit
def embedding_kernel(
    out_ptr,  # pointer to the output
    in_ptr,  # pointer to the input indices
    weight_ptr,  # pointer to the weights
    other_ptr,  # pointer to the other tensor
    N: tl.constexpr,  # number of columns in the embedding matrix
    BLOCK_SIZE: tl.constexpr,
    max_norm: tl.float32,  # maximum norm for renormalization
    norm_type: tl.float32,  # p-norm type
):
    pid = tl.program_id(0)
    out_ptr += pid * N
    in_ptr += pid

    mask = tl.arange(0, BLOCK_SIZE) < N
    cols = tl.arange(0, BLOCK_SIZE)

    row_idx = tl.load(in_ptr)
    weight_ptr += row_idx * N
    embedding_weight = tl.load(weight_ptr + cols, mask, other=0.0)

    # Renormalize if max_norm is provided
    if max_norm > 0:
        norm = tl.norm(embedding_weight, norm_type)
        if norm > max_norm:
            embedding_weight = embedding_weight * (max_norm / norm)

    other_value = tl.load(other_ptr + cols, mask, other=0.0)
    sum_value = embedding_weight + other_value
    tanh_value = tl.tanh(sum_value)
    tl.store(out_ptr + cols, tanh_value, mask)

@triton.jit
def embedding_backward_kernel(
    grad_in,  # pointer to the gradient input
    grad_out,  # pointer to the gradient output
    indices,  # pointer to the input indices
    other,  # pointer to the other tensor
    padding_idx,  # padding_idx
    max_norm: tl.float32,  # maximum norm for renormalization
    norm_type: tl.float32,  # p-norm type
    HAS_PADDING_IDX: tl.constexpr,
    N: tl.constexpr,  # number of columns in the embedding matrix
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    grad_out += pid * N
    indices += pid

    mask = tl.arange(0, BLOCK_SIZE) < N
    cols = tl.arange(0, BLOCK_SIZE)

    row_idx = tl.load(indices).to(tl.int32)
    if not HAS_PADDING_IDX or row_idx != padding_idx:
        grad_in += row_idx * N
        other_value = tl.load(other + cols, mask, other=0.0)
        embedding_grad = tl.load(grad_out + cols, mask, other=0.0)

        # Renormalize if max_norm is provided
        if max_norm > 0:
            norm = tl.norm(embedding_grad, norm_type)
            if norm > max_norm:
                embedding_grad = embedding_grad * (max_norm / norm)

        tl.atomic_add(grad_in + cols, embedding_grad, mask=mask)

class FusedEmbeddingAddTanh(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input_indices,
        weight,
        other,
        padding_idx=-1,
        max_norm=None,
        norm_type=2.0,
        scale_grad_by_freq=False,
        sparse=False,
        out=None
    ):
        M = input_indices.numel()
        N = weight.shape[-1]

        BLOCK_SIZE = triton.next_power_of_2(N)
        input_indices = input_indices.contiguous()
        weight = weight.contiguous()
        other = other.contiguous()

        if out is None:
            out = torch.empty(
                (*input_indices.shape, N), device=input_indices.device, dtype=weight.dtype
            )

        with torch.cuda.device(weight.device):
            embedding_kernel[M,](
                out, input_indices, weight, other, N, BLOCK_SIZE, max_norm or 0, norm_type
            )

        ctx.save_for_backward(input_indices, weight, other)
        ctx.padding_idx = padding_idx
        ctx.max_norm = max_norm
        ctx.norm_type = norm_type
        ctx.scale_grad_by_freq = scale_grad_by_freq
        ctx.sparse = sparse

        return out

    @staticmethod
    def backward(ctx, grad_outputs):
        input_indices, weight, other = ctx.saved_tensors
        grad_inputs = torch.zeros_like(weight)

        if ctx.scale_grad_by_freq:
            indice_freq = torch.zeros(
                (weight.shape[0],),
                requires_grad=False,
                device=grad_outputs.device,
                dtype=torch.int32,
            )
            INDICE_BLOCK_SIZE = 256
            indice_grid = lambda meta: (triton.cdiv(input_indices.numel(), INDICE_BLOCK_SIZE),)

            with torch.cuda.device(grad_outputs.device):
                indice_freq_kernel[indice_grid](
                    indice_freq, input_indices, input_indices.numel(), INDICE_BLOCK_SIZE
                )
        else:
            indice_freq = None

        BLOCK_SIZE = triton.next_power_of_2(weight.shape[-1])
        HAS_PADDING_IDX = ctx.padding_idx is not None

        with torch.cuda.device(grad_outputs.device):
            embedding_backward_kernel[input_indices.numel(),](
                grad_inputs,
                grad_outputs,
                input_indices,
                other,
                ctx.padding_idx,
                ctx.max_norm or 0,
                ctx.norm_type,
                HAS_PADDING_IDX,
                weight.shape[-1],
                BLOCK_SIZE,
            )

        if ctx.scale_grad_by_freq:
            with torch.cuda.device(grad_outputs.device):
                embedding_grad_scale_kernel[weight.shape[0],](
                    grad_inputs, indice_freq, weight.shape[0], weight.shape[-1], BLOCK_SIZE
                )

        return None, grad_inputs, None, None, None, None, None, None, None

def fused_embedding_add_tanh(
    input_indices,
    weight,
    other,
    *,
    padding_idx=None,
    max_norm=None,
    norm_type=2.0,
    scale_grad_by_freq=False,
    sparse=False,
    out=None
):
    return FusedEmbeddingAddTanh.apply(
        input_indices, weight, other, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse, out
    )
